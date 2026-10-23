import os
from pathlib import Path

import pytest
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def test_database_url() -> str:
    url = os.environ.get("CXR_TEST_DATABASE_URL")
    if not url:
        pytest.skip("CXR_TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def alembic_config(test_database_url: str) -> Config:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("sqlalchemy.url", test_database_url)
    config.attributes["configure_logger"] = False
    return config
