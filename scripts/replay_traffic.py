"""Replay NIH images through the public API, as production traffic for the monitor flow.

    uv run python scripts/replay_traffic.py [--n 150] [--rate 1.5] [--feedback 0.3]
    make replay ARGS="--n 300 --feedback 0.5"

Uploads images from the validation split (the split the drift reference was built from, so
a random sample shouldn't drift) with their sex, age, and view as form fields, the way a
PACS integration would send them. With `--feedback F`, a share F of the finished jobs gets
reviewer feedback with the image's NIH labels, which the monitor flow turns into delayed
AUROC. `--slice` sends only part of the split; `inject_drift.py` uses it for AP views.

Needs the stack up (`make up`), a `@champion` (`make seed-model`), and the images pulled
(`dvc pull data/nih`). Nginx limits uploads to 2/s, so keep `--rate` below that.
"""

import argparse
import csv
import operator
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
METADATA = REPO / "data" / "nih" / "metadata.csv"
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge, "<=": operator.le, "!=": operator.ne, "=": operator.eq,
    ">": operator.gt, "<": operator.lt,
}  # fmt: skip
_SLICE = re.compile(r"^(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)$")


def parse_slice(text: str) -> Callable[[dict[str, str]], bool]:
    """`view_position=AP`, `patient_age>=70`, `patient_sex!=M` → a row predicate."""
    match = _SLICE.match(text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"not a slice: {text!r} (e.g. view_position=AP)")
    column, op, raw = match.groups()

    def keep(row: dict[str, str]) -> bool:
        value = row.get(column, "")
        if value == "":
            return False
        try:
            return _OPS[op](float(value), float(raw))
        except ValueError:
            return _OPS[op](value, raw)

    return keep


def select(rows: Sequence[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    pool = [r for r in rows if r["split"] == args.split]
    for text in args.slice:
        keep = parse_slice(text)
        pool = [r for r in pool if keep(r)]
    if not pool:
        sys.exit(f"no {args.split} images match {args.slice}")
    rng = random.Random(args.seed)
    return rng.sample(pool, min(args.n, len(pool)))


def upload(client: httpx.Client, row: dict[str, str], send_metadata: bool) -> str | None:
    path = METADATA.parent / row["file"]
    fields = (
        {k: row[k] for k in ("patient_sex", "patient_age", "view_position") if row.get(k)}
        if send_metadata
        else {}
    )
    for attempt in range(6):
        with path.open("rb") as f:
            response = client.post(
                "/api/jobs", files={"file": (path.name, f, "image/png")}, data=fields
            )
        if response.status_code == 429:  # nginx rate limit: back off and retry
            time.sleep(0.5 * 2**attempt)
            continue
        if response.status_code != 202:
            print(f"  {row['image_id']}: HTTP {response.status_code} {response.text[:200]}")
            return None
        job_id: str = response.json()["id"]
        return job_id
    print(f"  {row['image_id']}: still rate-limited, skipped")
    return None


def wait_for(client: httpx.Client, job_ids: Sequence[str], timeout_s: float) -> dict[str, Any]:
    """Poll in upload order at about 10 requests/s, under nginx's 20/s API limit."""
    pending, done = list(job_ids), {}
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{pending[0]}")
        if response.status_code == 429:
            time.sleep(1)
            continue
        body = response.json()
        if body["status"] == "queued":
            time.sleep(1)  # jobs finish roughly in order, so the rest are queued too
            continue
        done[pending.pop(0)] = body
        time.sleep(0.1)
    if pending:
        print(f"{len(pending)} job(s) still queued after {timeout_s:.0f}s")
    return done


def send_feedback(client: httpx.Client, job: dict[str, Any], row: dict[str, str]) -> bool:
    classes = job["prediction"]["classes"]
    if any(c not in row for c in classes):
        return False
    labels = {c: row[c] == "1" for c in classes}
    response = client.post(
        f"/api/jobs/{job['id']}/feedback",
        json={"verdict": "correct", "labels": labels, "comment": "replay: NIH labels"},
    )
    return response.status_code == 201


def main(argv: Sequence[str] | None = None, *, default_slice: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--n", type=int, default=150, help="images to send (default 150)")
    parser.add_argument("--rate", type=float, default=1.5, help="uploads per second")
    parser.add_argument("--split", default="val", help="NIH split to sample (default val)")
    parser.add_argument(
        "--slice", action="append", default=list(default_slice),
        help="keep only matching rows, e.g. view_position=AP or patient_age>=70 (repeatable)",
    )  # fmt: skip
    parser.add_argument("--feedback", type=float, default=0.0, metavar="SHARE",
                        help="share of finished jobs to give NIH-label feedback (0-1)")  # fmt: skip
    parser.add_argument("--no-metadata", action="store_true",
                        help="don't send sex, age, and view with the upload")  # fmt: skip
    parser.add_argument("--seed", type=int, default=None, help="sample seed (default: random)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for jobs")
    args = parser.parse_args(argv)
    if not 0 <= args.feedback <= 1:
        parser.error("--feedback is a share between 0 and 1")

    if not METADATA.exists():
        sys.exit(f"{METADATA} is missing: run `make seed` or `dvc pull data/nih` first")
    with METADATA.open() as f:
        rows = list(csv.DictReader(f))
    chosen = select(rows, args)
    if not (METADATA.parent / chosen[0]["file"]).exists():
        sys.exit("the images aren't here: `dvc pull data/nih` first")
    by_job: dict[str, dict[str, str]] = {}
    slice_text = f" where {' and '.join(args.slice)}" if args.slice else ""
    print(f"sending {len(chosen)} {args.split} images{slice_text} at {args.rate}/s")

    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        interval = 1 / args.rate if args.rate > 0 else 0
        for i, row in enumerate(chosen, 1):
            started = time.monotonic()
            job_id = upload(client, row, send_metadata=not args.no_metadata)
            if job_id:
                by_job[job_id] = row
            if i % 25 == 0 or i == len(chosen):
                print(f"  {i}/{len(chosen)} uploaded")
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

        jobs = wait_for(client, list(by_job), args.timeout)
        ok = [j for j in jobs.values() if j["status"] == "succeeded"]
        versions = sorted({j["prediction"]["model_version"] for j in ok})
        print(f"{len(ok)} succeeded, {len(jobs) - len(ok)} failed, model version(s) {versions}")

        if args.feedback and ok:
            rng = random.Random(args.seed)
            picked = [j for j in ok if rng.random() < args.feedback]
            sent = sum(send_feedback(client, j, by_job[j["id"]]) for j in picked)
            print(f"feedback with NIH labels on {sent} job(s)")


if __name__ == "__main__":
    main()
