"""Run the real migrations against a throwaway Postgres database.

Destructive: drops everything in the target database's managed schemas.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from cxr.db import Base
from cxr.db.base import SCHEMAS
from cxr.labels import CLASSES

pytestmark = pytest.mark.db


@pytest.fixture
def engine(test_database_url):
    engine = create_engine(test_database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def migrated(alembic_config, engine):
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    yield engine
    command.downgrade(alembic_config, "base")


def test_upgrade_creates_every_table(migrated):
    inspector = inspect(migrated)
    actual = {
        f"{schema}.{table}" for schema in SCHEMAS for table in inspector.get_table_names(schema)
    }

    assert actual == set(Base.metadata.tables)


def test_models_match_migrations(migrated):
    with migrated.connect() as conn:
        context = MigrationContext.configure(
            conn,
            opts={
                "include_schemas": True,
                "include_name": lambda name, type_, _: type_ != "schema" or name in SCHEMAS,
                "compare_server_default": True,
            },
        )
        diff = compare_metadata(context, Base.metadata)

    assert diff == []


def test_findings_are_seeded(migrated):
    with migrated.connect() as conn:
        rows = conn.execute(
            text("SELECT finding_id, name FROM dataset.findings ORDER BY finding_id")
        ).all()

    assert [tuple(row) for row in rows] == list(enumerate(CLASSES))


def test_downgrade_removes_schemas(alembic_config, migrated):
    command.downgrade(alembic_config, "base")

    remaining = set(inspect(migrated).get_schema_names()) & set(SCHEMAS)
    assert remaining == set()
