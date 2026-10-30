"""Names and message shapes on RabbitMQ, shared by the backend (publisher) and the worker.

Stdlib only. The exchanges and the dead-letter policy are created by
compose/rabbitmq/definitions.json; the `inference` queue is declared by Celery and picks up
the `^inference` dead-letter policy, so rejected jobs land in `dlq`.
"""

import json
from dataclasses import asdict, dataclass

INFERENCE_QUEUE = "inference"
INFER_TASK = "cxr_worker.infer"
# Fanout exchange; every worker binds its own exclusive queue to it.
MODEL_EVENTS_EXCHANGE = "model.events"


@dataclass(frozen=True)
class ChampionChanged:
    """Published on `model.events` whenever `@champion` moves. Workers load `bundle_uri`."""

    model_name: str
    model_version: str
    bundle_uri: str
    event: str = "champion_changed"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_payload(cls, payload: object) -> ChampionChanged:
        """Accepts the decoded JSON body (dict) or the raw string; raises ValueError otherwise."""
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
        if not isinstance(data, dict) or data.get("event") != "champion_changed":
            raise ValueError(f"not a champion_changed event: {payload!r}")
        try:
            return cls(
                model_name=str(data["model_name"]),
                model_version=str(data["model_version"]),
                bundle_uri=str(data["bundle_uri"]),
            )
        except KeyError as exc:
            raise ValueError(f"champion_changed event without {exc}") from None
