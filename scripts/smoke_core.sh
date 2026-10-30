#!/bin/sh
# Core smoke test: upload an image through nginx, wait for the worker, and check that the
# prediction names the current @champion version and that its heatmap is served.
# Usage: scripts/smoke_core.sh [image]   (needs `make up` and `make seed-model` first)
set -eu

base=${BASE_URL:-http://localhost:8000}
mlflow=${MLFLOW_URL:-http://localhost:5000}
image=${1:-data/golden/png/00000001_000.png}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "smoke: FAIL: $*" >&2; exit 1; }
field() { python3 -c "import json, sys; d = json.load(sys.stdin); print($1)"; }

if [ ! -f "$image" ]; then
    # The golden set lives in DVC, so fresh clones and CI don't have it. A synthetic PNG
    # takes the same path through the stack; this test checks plumbing, not accuracy.
    image=$tmp/synthetic.png
    python3 - "$image" <<'PY'
import struct, sys, zlib
w = h = 256
rows = b"".join(b"\0" + bytes((x + y) // 2 for x in range(w)) for y in range(h))
def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PY
    echo "smoke: no golden image here, using a synthetic one"
fi

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
