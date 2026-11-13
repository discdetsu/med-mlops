#!/bin/sh
# Core smoke test: upload an image through nginx, wait for the worker, and check that the
# prediction names the current @champion version and that its heatmap is served.
# Usage: scripts/smoke_core.sh [image]   (needs `make up` and `make seed-model` first)
set -eu

base=${BASE_URL:-http://localhost:8000}
mlflow=${MLFLOW_URL:-http://localhost:5000}
image=${1:-docs/samples/00019087_046_effusion.png}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "smoke: FAIL: $*" >&2; exit 1; }
field() { python3 -c "import json, sys; d = json.load(sys.stdin); print($1)"; }

[ -f "$image" ] || fail "no image at $image"

champion=$(curl -fsS "$mlflow/api/2.0/mlflow/registered-models/alias?name=cxr-classifier&alias=champion" \
    | field 'd["model_version"]["version"]') || fail "no @champion in MLflow; run make seed-model"

id=$(curl -fsS -F "file=@$image" "$base/api/jobs" | field 'd["id"]') || fail "upload rejected"
echo "smoke: job $id queued"

status=queued
for _ in $(seq 1 60); do
    curl -fsS "$base/api/jobs/$id" > "$tmp/job.json"
    status=$(field 'd["status"]' < "$tmp/job.json")
    [ "$status" != queued ] && break
    sleep 1
done
[ "$status" = succeeded ] || fail "job ended as '$status': $(field 'd["error"]' < "$tmp/job.json")"

version=$(field 'd["prediction"]["model_version"]' < "$tmp/job.json")
[ "$version" = "$champion" ] || fail "prediction from version $version, @champion is $champion"

heatmap=$(field 'next(iter(d["prediction"]["heatmap_urls"].values()))' < "$tmp/job.json")
curl -fsS "$base$heatmap" -o "$tmp/cam.png"
[ "$(head -c 4 "$tmp/cam.png" | tail -c 3)" = PNG ] || fail "heatmap is not a PNG"

field '{c: round(p, 3) for c, p in d["prediction"]["probabilities"].items()}' < "$tmp/job.json"
echo "smoke: OK, model version $version"
