from django.conf import settings
from kombu import Connection

from cxr_core.messaging import MODEL_EVENTS_EXCHANGE, ChampionChanged


def publish_champion_changed(event: ChampionChanged) -> None:
    """Fan the event out to every worker. The exchange is created by rabbitmq-init."""
    with Connection(settings.CELERY_BROKER_URL, connect_timeout=5) as conn:
        conn.Producer().publish(
            event.to_json(),
            exchange=MODEL_EVENTS_EXCHANGE,
            routing_key="",
            content_type="application/json",
            content_encoding="utf-8",
            retry=True,
            retry_policy={"max_retries": 3},
        )
