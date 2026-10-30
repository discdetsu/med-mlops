"""`model.events` subscriber: one exclusive, server-named queue per worker on the fanout
exchange, so every worker hears every event."""

import logging
from typing import Any

from celery import bootsteps
from kombu import Consumer, Exchange, Message, Queue

from cxr_core.messaging import MODEL_EVENTS_EXCHANGE, ChampionChanged
from cxr_worker.backend import ChampionInfo
from cxr_worker.context import context

logger = logging.getLogger(__name__)


class ModelEvents(bootsteps.ConsumerStep):  # type: ignore[misc]
    def get_consumers(self, channel: Any) -> list[Consumer]:
        # Runs on startup and after every reconnect. Events sent while disconnected are lost
        # with the old exclusive queue, so catch up with the registry first.
        context().holder.sync()
        queue = Queue(
            exchange=Exchange(MODEL_EVENTS_EXCHANGE, type="fanout", passive=True),
            exclusive=True,
            auto_delete=True,
        )
        return [Consumer(channel, queues=[queue], callbacks=[on_event], accept=["json"])]


def on_event(body: Any, message: Message) -> None:
    """Runs in the worker's event loop, so never in the middle of a job (solo pool)."""
    try:
        event = ChampionChanged.from_payload(body)
        context().holder.load(ChampionInfo(event.model_name, event.model_version, event.bundle_uri))
    except Exception:
        logger.exception("ignoring model event %r", body)
    finally:
        message.ack()
