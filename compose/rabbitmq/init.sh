#!/bin/sh
# Imports exchanges, queues, and policies through the management API. Importing at
# boot (definitions.local.path) would skip creating the default user, so it runs here.
# Idempotent: re-importing existing definitions is a no-op.
set -eu

auth=$(printf '%s:%s' "$RABBITMQ_USER" "$RABBITMQ_PASSWORD" | base64)
wget -qO- \
    --header "Authorization: Basic $auth" \
    --header "Content-Type: application/json" \
    --post-file /init/definitions.json \
    http://rabbitmq:15672/api/definitions
echo "definitions imported"
