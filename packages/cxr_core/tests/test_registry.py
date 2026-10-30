import json

import numpy as np
import pytest

from cxr_core.inference import run_onnx, sigmoid
from cxr_core.messaging import ChampionChanged
from cxr_core.registry import bundle_prefix, bundle_uri, split_s3_uri


def test_bundle_uri_round_trips_through_split() -> None:
    uri = bundle_uri("cxr-classifier", "3")
    assert uri == "s3://artifacts/models/cxr-classifier/3/"
    assert split_s3_uri(uri) == ("artifacts", bundle_prefix("cxr-classifier", "3"))


@pytest.mark.parametrize("uri", ["artifacts/models/1/", "s3:///models/1/", "http://x/y"])
def test_split_rejects_non_s3_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        split_s3_uri(uri)


def test_champion_changed_round_trips() -> None:
    event = ChampionChanged("cxr-classifier", "4", "s3://artifacts/models/cxr-classifier/4/")
    assert ChampionChanged.from_payload(event.to_json()) == event
    assert ChampionChanged.from_payload(json.loads(event.to_json())) == event


@pytest.mark.parametrize(
    "payload",
    [
        {"event": "something_else", "model_name": "m", "model_version": "1", "bundle_uri": "u"},
        {"event": "champion_changed", "model_name": "m"},
        ["not", "a", "dict"],
        "not json",
    ],
)
def test_malformed_events_raise_value_error(payload: object) -> None:
    with pytest.raises(ValueError):
        ChampionChanged.from_payload(payload)


def test_run_onnx_asks_for_both_outputs_by_name() -> None:
    class Session:
        def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert output_names == ["logits", "features"]
            assert set(feed) == {"image"}
            return [np.zeros((1, 2)), np.ones((1, 3, 2, 2))]

    logits, features = run_onnx(Session(), np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert logits.dtype == features.dtype == np.float32
    np.testing.assert_allclose(sigmoid(logits), 0.5)
