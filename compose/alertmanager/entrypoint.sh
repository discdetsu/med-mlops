#!/bin/sh
# Alertmanager reads no env vars in its config, so pick the receiver here and hand the webhook
# URL over as a file (webhook_url_file / api_url_file) instead of writing it into the config.
set -eu

dir=/tmp/alertmanager
mkdir -p "$dir"
printf '%s' "${DISCORD_WEBHOOK_URL:-}" > "$dir/discord_webhook"
printf '%s' "${SLACK_WEBHOOK_URL:-}" > "$dir/slack_webhook"
chmod 600 "$dir/discord_webhook" "$dir/slack_webhook"

if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    receiver=discord
elif [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    receiver=slack
else
    receiver=ui-only
fi
sed "s/__RECEIVER__/$receiver/" /etc/alertmanager/alertmanager.yml > "$dir/alertmanager.yml"
echo "alertmanager: sending alerts to '$receiver'"
exec /bin/alertmanager --config.file="$dir/alertmanager.yml" "$@"
