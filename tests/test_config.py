import pytest
from pydantic import ValidationError

from cxr.config import Settings

REQUIRED = {
    "CXR_DB_PASSWORD": "db-secret",
    "CXR_S3_ACCESS_KEY": "access",
    "CXR_S3_SECRET_KEY": "s3-secret",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def test_defaults_point_at_local_compose(env):
    settings = Settings(_env_file=None)

    assert settings.database_url.render_as_string(hide_password=False) == (
        "postgresql+psycopg://app:db-secret@localhost:5432/app"
    )
    assert settings.s3_endpoint_url == "http://localhost:9000"
    assert settings.s3_bucket == "cxr"


def test_env_overrides(env):
    env.setenv("CXR_DB_HOST", "postgres")
    env.setenv("CXR_DB_PORT", "6543")

    url = Settings(_env_file=None).database_url

    assert (url.host, url.port) == ("postgres", 6543)


def test_secrets_are_not_printed(env):
    text = repr(Settings(_env_file=None))

    assert "db-secret" not in text
    assert "s3-secret" not in text


def test_missing_secret_fails(env):
    env.delenv("CXR_DB_PASSWORD")

    with pytest.raises(ValidationError, match="db_password"):
        Settings(_env_file=None)
