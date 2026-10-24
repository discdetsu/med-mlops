import os
import time
import uuid

import pytest
from kombu import Connection, Exchange, Queue

pytestmark = pytest.mark.integration


def test_model_events_fans_out_to_every_worker() -> None:
    """Each worker binds its own exclusive queue; one event must reach all of them."""
    exchange = Exchange("model.events", type="fanout", passive=True)
    event = {"model_version": uuid.uuid4().hex}

    with Connection(os.environ["RABBITMQ_URL"]) as conn:
        queues = [
            Queue(exchange=exchange, exclusive=True, auto_delete=True)(conn.default_channel)
            for _ in range(2)
        ]
        for q in queues:
            q.declare()

        conn.Producer().publish(event, exchange=exchange, serializer="json")

        for q in queues:
            msg = None
            for _ in range(20):
                msg = q.get(no_ack=True)
                if msg:
                    break
                time.sleep(0.1)
            assert msg is not None
            assert msg.payload == event


def test_dead_letter_queue_exists() -> None:
    with Connection(os.environ["RABBITMQ_URL"]) as conn:
        # passive: fails if the queue is missing, never creates or redeclares it
        conn.default_channel.queue_declare(queue="dlq", passive=True)
