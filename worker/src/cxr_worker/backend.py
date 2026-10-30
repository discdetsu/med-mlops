"""The backend's internal API: the champion lookup and result reports."""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ChampionInfo:
    model_name: str
    model_version: str
    bundle_uri: str


class NoChampion(Exception):
    """The registry has no `@champion` yet; `make seed-model` hasn't run."""


class BackendClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 10.0) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )

    def champion(self) -> ChampionInfo:
        response = self._http.get("/internal/champion")
        if response.status_code == 404:
            raise NoChampion(response.json().get("detail", "no champion"))
        response.raise_for_status()
        data = response.json()
        return ChampionInfo(data["model_name"], data["model_version"], data["bundle_uri"])

    def post_result(self, payload: dict[str, Any]) -> None:
        self._http.post("/internal/results", json=payload).raise_for_status()
