import time
from pathlib import Path
from typing import Any

import pytest
from kombu import Message

import cxr_worker.events as events
import cxr_worker.tasks as tasks
from cxr_worker.context import Context
from cxr_worker.deadline import JobTimeout, deadline
from cxr_worker.model import ModelHolder
from cxr_worker.settings import Settings

from .conftest import BundleFactory, FakeS3
from .fakes import FakeBackend, info

JOB = "6f1c7a52-0000-0000-0000-000000000001"


@pytest.fixture
def ctx(
    tmp_path: Path,
    s3: FakeS3,
    make_bundle: BundleFactory,
    golden_png: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Context:
    backend = FakeBackend(info("1"))
    for v in ("1", "2"):
        s3.add_bundle(info(v).bundle_uri, make_bundle(v))
    s3.objects[("inference", f"raw/{JOB}.png")] = golden_png.read_bytes()
    settings = Settings(
        broker_url="memory://",
        backend_url="http://backend",
        internal_token="t",
        model_cache=tmp_path / "cache",
        job_timeout_s=10,
        display_size=32,
    )
    context = Context(settings, s3, backend, ModelHolder(settings.model_cache, s3, backend))  # type: ignore[arg-type]
    monkeypatch.setattr(tasks, "context", lambda: context)
    monkeypatch.setattr(events, "context", lambda: context)
    # Eager retries would otherwise wait out their countdown in the test.
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return context


def run(job_id: str = JOB, raw_key: str | None = None, **request: Any) -> Any:
    return tasks.infer.apply(args=[job_id, raw_key or f"raw/{job_id}.png"], **request)


def results(ctx: Context) -> list[dict[str, Any]]:
    return ctx.backend.results  # type: ignore[attr-defined, no-any-return]


def test_job_reports_prediction_with_lineage_and_uploads_pngs(ctx: Context) -> None:
    result = run()
    assert result.successful(), result.traceback

    [report] = results(ctx)
    assert report["job_id"] == JOB
    assert report["status"] == "succeeded"
    p = report["prediction"]
    assert p["model_version"] == "1"
    assert p["mlflow_run_id"] == "run-1"
    assert p["dataset_rev"] == "rev1"
    assert p["thresholds"] == {"Atelectasis": 0.5, "Effusion": 0.4}
    assert set(p["probabilities"]) == {"Atelectasis", "Effusion"}
    assert p["latency_ms"] > 0
    assert p["image_key"] == f"results/{JOB}/image.png"
    assert p["heatmap_keys"] == {
        "Atelectasis": f"results/{JOB}/cam/0.png",
        "Effusion": f"results/{JOB}/cam/1.png",
    }
    s3: FakeS3 = ctx.s3
    for key in [p["image_key"], *p["heatmap_keys"].values()]:
        assert s3.objects[("inference", key)].startswith(b"\x89PNG")


def test_undecodable_image_fails_the_job_without_retrying(ctx: Context) -> None:
    ctx.s3.objects[("inference", "raw/bad.png")] = b"\x89PNG\r\n\x1a\nnot really"
    result = run(raw_key="raw/bad.png")

    assert result.failed()
    assert isinstance(result.result, tasks.PoisonJob)
    [report] = results(ctx)
    assert report["status"] == "failed"
    assert report["error"] == "OSError: Truncated File Read"


def test_missing_raw_file_fails_the_job(ctx: Context) -> None:
    result = run(raw_key="raw/gone.png")
    assert isinstance(result.result, tasks.PoisonJob)
    assert results(ctx)[0]["error"] == "the uploaded file is gone from storage"


def test_redelivered_job_is_treated_as_poison(ctx: Context) -> None:
    # Eager apply() can't mark a delivery as redelivered; set the request a broker would.
    tasks.infer.push_request(delivery_info={"redelivered": True}, retries=0)
    try:
        with pytest.raises(tasks.PoisonJob):
            tasks.infer.run(JOB, f"raw/{JOB}.png")
    finally:
        tasks.infer.pop_request()
    assert results(ctx)[0]["status"] == "failed"


def test_backend_outage_is_retried(ctx: Context) -> None:
    ctx.backend.fail_posts = 2  # type: ignore[attr-defined]
    result = run()
    assert result.successful(), result.traceback
    assert [r["status"] for r in results(ctx)] == ["succeeded"]


def test_no_champion_retries_then_fails(ctx: Context) -> None:
    ctx.backend.champion_info = None  # type: ignore[attr-defined]
    result = run()

    assert result.failed()
    [report] = results(ctx)
    assert report["status"] == "failed"
    assert report["error"].startswith(f"gave up after {tasks.MAX_RETRIES} retries")


def test_model_event_swaps_the_model_between_jobs(ctx: Context) -> None:
    run()
    acked: list[bool] = []
    message = Message(body=b"", channel=None)
    message.ack = lambda: acked.append(True)  # type: ignore[method-assign]

    events.on_event(
        {
            "event": "champion_changed",
            "model_name": "cxr-classifier",
            "model_version": "2",
            "bundle_uri": info("2").bundle_uri,
        },
        message,
    )
    run()

    assert acked == [True]
    assert [r["prediction"]["model_version"] for r in results(ctx)] == ["1", "2"]


def test_malformed_model_event_is_acked_and_ignored(ctx: Context) -> None:
    run()
    acked: list[bool] = []
    message = Message(body=b"", channel=None)
    message.ack = lambda: acked.append(True)  # type: ignore[method-assign]

    events.on_event({"event": "something_else"}, message)

    assert acked == [True]
    assert ctx.holder.current is not None
    assert ctx.holder.current.version == "1"


def test_deadline_interrupts_a_stuck_job() -> None:
    with pytest.raises(JobTimeout), deadline(0.05):
        while True:
            pass
