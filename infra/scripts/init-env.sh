#!/usr/bin/env bash
# Create `.env` from `.env.example`, replacing every `change-me` with a random secret.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -e .env ]]; then
    echo ".env already exists; not overwriting" >&2
    exit 1
fi

python3 - <<'EOF'
import base64
import os
import secrets
from pathlib import Path

lines = []
for line in Path(".env.example").read_text().splitlines():
    key, sep, value = line.partition("=")
    if sep and value == "change-me":
        if key == "AIRFLOW_FERNET_KEY":
            value = base64.urlsafe_b64encode(os.urandom(32)).decode()
        else:
            value = secrets.token_urlsafe(24)
    lines.append(f"{key}{sep}{value}")
Path(".env").write_text("\n".join(lines) + "\n")
EOF

chmod 600 .env
echo "wrote .env"
