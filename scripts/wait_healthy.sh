#!/bin/sh
# Wait until every service in a compose profile is healthy, or exited 0 for one-shot
# `*-init` jobs. `docker compose up --wait` treats any exited container as a failure.
# Usage: scripts/wait_healthy.sh <profile> [timeout_seconds]
set -eu

profile=${1:?usage: wait_healthy.sh <profile> [timeout_seconds]}
deadline=$(( $(date +%s) + ${2:-300} ))

while :; do
    status=$(docker compose --profile "$profile" ps -a --format '{{.Service}}|{{.State}}|{{.Health}}|{{.ExitCode}}')
    failed=$(echo "$status" | awk -F'|' '($2 == "exited" && ($4 != 0 || $1 !~ /-init$/)) || $3 == "unhealthy"')
    pending=$(echo "$status" | awk -F'|' '!(($2 == "running" && $3 == "healthy") || ($2 == "exited" && $4 == 0 && $1 ~ /-init$/))')

    if [ -n "$failed" ]; then
        echo "failed:" >&2; echo "$failed" >&2; exit 1
    fi
    if [ -z "$pending" ]; then
        echo "all '$profile' services are healthy"; exit 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "timed out waiting for:" >&2; echo "$pending" >&2; exit 1
    fi
    sleep 3
done
