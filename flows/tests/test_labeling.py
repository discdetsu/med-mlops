"""The labeling loop's decisions: selection, answers, splits, and should_retrain."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cxr_core.registry import REASON_DISAGREEMENT, REASON_DRIFT, REASON_RANDOM
from cxr_flows import labeling
from cxr_flows.config import LabelingConfig, load_labeling_config
from cxr_flows.labelstudio import NONE_CHOICE, choices_result
from cxr_flows.nih import patient_split

CLASSES = ("Effusion", "Mass")


def _patient(bucket: str, start: int = 0) -> str:
    """A patient ID that the split hash puts in `bucket`."""
    for i in range(start, start + 10_000):
        if patient_split(f"{i:08d}") == bucket:
            return f"{i:08d}"
    raise AssertionError(bucket)


@pytest.fixture
def cfg() -> LabelingConfig:
    return load_labeling_config().model_copy(
        update={"top_k_drift": 2, "random_k": 1, "max_tasks_per_pass": 100}
    )


def _feedback(rows: list[tuple[str, dict[str, int], dict[str, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [f"job-{sha}" for sha, _, _ in rows],
            "model_version": "2",
            "labels": [labels for _, labels, _ in rows],
            "probabilities": [probs for _, _, probs in rows],
            "thresholds": [dict.fromkeys(CLASSES, 0.5)] * len(rows),
            "patient_id": _patient("train"),
            "sha256": [sha for sha, _, _ in rows],
            "image_key": [f"results/{sha}/image.png" for sha, _, _ in rows],
            "created_at": pd.date_range("2026-11-01", periods=len(rows), freq="h"),
        }
    )


def _window(embeddings: list[list[float]], prefix: str = "w") -> pd.DataFrame:
    n = len(embeddings)
    return pd.DataFrame(
        {
            "job_id": [f"job-{prefix}{i}" for i in range(n)],
            "model_version": "2",
            "embedding": embeddings,
            "patient_id": _patient("train"),
            "sha256": [f"{prefix}{i}" for i in range(n)],
            "image_key": [f"results/{prefix}{i}/image.png" for i in range(n)],
        }
    )


def _reference(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 1, (n, 2)), columns=["emb_0000", "emb_0001"])


# ------------------------------------------------------------------ curated split


def test_test_locked_patients_are_refused_and_val_patients_never_train() -> None:
    assert labeling.curated_split(_patient("test_locked")) == "test_locked"
    assert labeling.curated_split(_patient("val")) == "recent"
    assert labeling.curated_split(_patient("val", 500)) == "recent"


def test_train_patients_split_about_as_configured_and_stay_put() -> None:
    patients = [f"{i:08d}" for i in range(4000) if patient_split(f"{i:08d}") == "train"]
    splits = [labeling.curated_split(p) for p in patients]
    assert set(splits) == {"train", "recent"}
    assert 0.15 < splits.count("recent") / len(splits) < 0.25
    assert splits == [labeling.curated_split(p) for p in patients]


def test_a_case_is_labelable_only_if_its_label_could_enter_the_data() -> None:
    assert labeling.is_labelable(_patient("train"))
    assert labeling.is_labelable(_patient("val"))
    assert not labeling.is_labelable(_patient("test_locked"))
    assert not labeling.is_labelable("")


# ------------------------------------------------------------------ selection


def test_disagreements_compare_labels_with_thresholded_predictions() -> None:
    feedback = _feedback(
        [
            ("same", {"Effusion": 1, "Mass": 0}, {"Effusion": 0.9, "Mass": 0.1}),
            ("missed", {"Effusion": 1, "Mass": 0}, {"Effusion": 0.2, "Mass": 0.1}),
            ("false-positive", {"Effusion": 0, "Mass": 0}, {"Effusion": 0.1, "Mass": 0.9}),
            ("at-threshold", {"Effusion": 1, "Mass": 0}, {"Effusion": 0.5, "Mass": 0.1}),
        ]
    )
    # Newest first.
    assert list(labeling.disagreements(feedback)["sha256"]) == ["false-positive", "missed"]


def test_farthest_ranks_by_distance_from_the_reference() -> None:
    rows = _window([[0.0, 0.0], [9.0, 9.0], [3.0, 3.0]])
    assert list(labeling.farthest(rows, _reference(), 2)["sha256"]) == ["w1", "w2"]


def test_distance_is_in_reference_standard_deviations() -> None:
    reference = pd.DataFrame({"emb_0000": [-100.0, 100.0], "emb_0001": [-1.0, 1.0]})
    near_wide, near_narrow = [[50.0, 0.0]], [[0.0, 5.0]]
    wide = labeling.embedding_distance(np.array(near_wide), reference)
    narrow = labeling.embedding_distance(np.array(near_narrow), reference)
    assert narrow[0] > wide[0]  # 5 std in the narrow dimension beats 0.5 std in the wide one


def test_a_flat_reference_dimension_does_not_divide_by_zero() -> None:
    reference = pd.DataFrame({"emb_0000": [1.0, 1.0], "emb_0001": [0.0, 2.0]})
    assert np.isfinite(labeling.embedding_distance(np.array([[5.0, 1.0]]), reference)).all()


def test_random_sample_is_the_same_for_the_same_window() -> None:
    rows = _window([[float(i), 0.0] for i in range(50)])
    first = labeling.random_sample(rows, 5)
    assert list(first["sha256"]) == list(labeling.random_sample(rows, 5)["sha256"])
    other = _window([[float(i), 0.0] for i in range(50)], prefix="x")
    assert [s[1:] for s in first["sha256"]] != [
        s[1:] for s in labeling.random_sample(other, 5)["sha256"]
    ]


def test_select_cases_takes_all_three_sources_in_order(cfg: LabelingConfig) -> None:
    feedback = _feedback([("dis", {"Effusion": 1, "Mass": 0}, {"Effusion": 0.1, "Mass": 0.1})])
    rows = _window([[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [8.0, 8.0], [0.2, 0.0]])
    window = labeling.NewWindow(rows, True, _reference())
    cases = labeling.select_cases(feedback, [window], set(), cfg)

    assert [c.reason for c in cases] == [
        REASON_DISAGREEMENT, REASON_DRIFT, REASON_DRIFT, REASON_RANDOM,
    ]  # fmt: skip
    assert [c.sha256 for c in cases[:3]] == ["dis", "w2", "w3"]
    assert cases[0].image_key == "results/dis/image.png"
    assert cases[3].sha256 in {"w0", "w1", "w4"}


def test_a_window_without_drift_only_gives_the_random_sample(cfg: LabelingConfig) -> None:
    rows = _window([[0.0, 0.0], [9.0, 9.0], [8.0, 8.0]])
    cases = labeling.select_cases(
        _feedback([]), [labeling.NewWindow(rows, False, _reference())], set(), cfg
    )
    assert [c.reason for c in cases] == [REASON_RANDOM]


def test_select_cases_skips_known_files_and_duplicates(cfg: LabelingConfig) -> None:
    disagree = {"Effusion": 1, "Mass": 0}, {"Effusion": 0.1, "Mass": 0.1}
    feedback = _feedback([("old", *disagree), ("new", *disagree), ("new", *disagree)])
    cases = labeling.select_cases(feedback, [], {"old"}, cfg)
    assert [c.sha256 for c in cases] == ["new"]


def test_select_cases_leaves_out_what_could_never_enter_the_data(cfg: LabelingConfig) -> None:
    disagree = {"Effusion": 1, "Mass": 0}, {"Effusion": 0.1, "Mass": 0.1}
    feedback = _feedback([("no-id", *disagree), ("locked", *disagree), ("good", *disagree)])
    feedback["patient_id"] = ["", _patient("test_locked"), _patient("train")]
    assert [c.sha256 for c in labeling.select_cases(feedback, [], set(), cfg)] == ["good"]


def test_the_cap_trims_disagreements_and_never_the_sample(cfg: LabelingConfig) -> None:
    """Seen live: 96 disagreements filled a cap of 100 and left no random sample."""
    capped = cfg.model_copy(update={"max_tasks_per_pass": 4})
    disagree = {"Effusion": 1, "Mass": 0}, {"Effusion": 0.1, "Mass": 0.1}
    feedback = _feedback([(f"d{i}", *disagree) for i in range(10)])
    rows = _window([[9.0, 9.0], [8.0, 8.0], [0.0, 0.0], [0.1, 0.1]])
    window = labeling.NewWindow(rows, True, _reference())
    cases = labeling.select_cases(feedback, [window], set(), capped)

    assert [c.reason for c in cases] == [
        REASON_DISAGREEMENT, REASON_DRIFT, REASON_DRIFT, REASON_RANDOM,
    ]  # fmt: skip
    assert cases[0].sha256 == "d9"  # the newest disagreement; the rest wait for the next pass


def test_a_cap_smaller_than_the_quotas_still_holds(cfg: LabelingConfig) -> None:
    capped = cfg.model_copy(update={"max_tasks_per_pass": 1})
    rows = _window([[9.0, 9.0], [8.0, 8.0], [0.0, 0.0]])
    window = labeling.NewWindow(rows, True, _reference())
    assert len(labeling.select_cases(_feedback([]), [window], set(), capped)) == 1


# ------------------------------------------------------------------ answers


def _annotation(choices: list[str], **fields: Any) -> dict[str, Any]:
    return {
        "id": fields.pop("id", 1),
        "result": choices_result(choices),
        "was_cancelled": False,
        "updated_at": "2026-11-01T00:00:00Z",
        **fields,
    }


def test_choices_become_one_label_per_class() -> None:
    assert labeling.parse_answer([_annotation(["Mass"])], CLASSES) == {"Effusion": 0, "Mass": 1}


def test_none_of_these_is_all_negative() -> None:
    answer = labeling.parse_answer([_annotation([NONE_CHOICE])], CLASSES)
    assert answer == {"Effusion": 0, "Mass": 0}


def test_the_latest_finished_annotation_wins() -> None:
    annotations = [
        _annotation(["Mass"], id=1, updated_at="2026-11-01T00:00:00Z"),
        _annotation(["Effusion"], id=2, updated_at="2026-11-02T00:00:00Z"),
        _annotation([NONE_CHOICE], id=3, updated_at="2026-11-03T00:00:00Z", was_cancelled=True),
    ]
    assert labeling.parse_answer(annotations, CLASSES) == {"Effusion": 1, "Mass": 0}


@pytest.mark.parametrize(
    "annotations",
    [
        [],
        [_annotation(["Mass"], was_cancelled=True)],  # skipped
        [{"id": 1, "result": [], "was_cancelled": False}],  # submitted with no answer
        [_annotation([])],
        [_annotation([NONE_CHOICE, "Mass"])],  # contradicts itself
        [_annotation(["Nodule"])],  # a choice this code doesn't know
    ],
)
def test_no_usable_answer_leaves_the_task_open(annotations: list[dict[str, Any]]) -> None:
    assert labeling.parse_answer(annotations, CLASSES) is None


# ------------------------------------------------------------------ should_retrain


def _decide(**overrides: Any) -> labeling.RetrainDecision:
    args = {
        "dataset_rev": "new",
        "baseline_rev": "old",
        "last_triggered_rev": "old",
        "new_labels": 50,
        "min_new_labels": 50,
    }
    return labeling.should_retrain(**(args | overrides))


def test_retrains_on_a_new_rev_with_enough_new_labels() -> None:
    decision = _decide()
    assert decision.trigger and "50 new labels" in decision.reason


def test_too_few_new_labels_wait() -> None:
    decision = _decide(new_labels=49)
    assert not decision.trigger and "at least 50" in decision.reason


def test_the_same_rev_as_the_newest_version_never_retrains() -> None:
    assert not _decide(dataset_rev="old", new_labels=500).trigger


def test_a_rev_is_only_started_once() -> None:
    decision = _decide(last_triggered_rev="new")
    assert not decision.trigger and "already started" in decision.reason


def test_a_failed_retrain_does_not_lose_its_labels() -> None:
    """A later rev counts from the newest registered version, so it can trigger again."""
    assert _decide(dataset_rev="newer", last_triggered_rev="new", new_labels=60).trigger


def test_no_model_yet_still_needs_enough_labels() -> None:
    assert _decide(baseline_rev="", last_triggered_rev=None).trigger
    assert not _decide(baseline_rev="", last_triggered_rev=None, new_labels=1).trigger
