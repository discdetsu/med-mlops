from typing import Any

from cxr_worker.backend import ChampionInfo, NoChampion


class FakeBackend:
    def __init__(self, champion: ChampionInfo | None = None) -> None:
        self.champion_info = champion
        self.results: list[dict[str, Any]] = []
        self.fail_posts = 0

    def champion(self) -> ChampionInfo:
        if self.champion_info is None:
            raise NoChampion("no @champion yet")
        return self.champion_info

    def post_result(self, payload: dict[str, Any]) -> None:
        import httpx

        if self.fail_posts:
            self.fail_posts -= 1
            raise httpx.ConnectError("backend down")
        self.results.append(payload)


def info(version: str) -> ChampionInfo:
    return ChampionInfo(
        "cxr-classifier", version, f"s3://artifacts/models/cxr-classifier/{version}/"
    )
