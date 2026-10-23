import importlib.util

from conftest import ROOT

from cxr.db import Base
from cxr.db.base import SCHEMAS
from cxr.labels import CLASSES, NO_FINDING


def _load_initial_migration():
    (path,) = (ROOT / "migrations" / "versions").glob("*-0001_*.py")
    spec = importlib.util.spec_from_file_location("migration_0001", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_table_lives_in_a_known_schema():
    schemas = {table.schema for table in Base.metadata.tables.values()}

    assert schemas == set(SCHEMAS)


def test_expected_tables():
    assert set(Base.metadata.tables) == {
        "dataset.etl_runs",
        "dataset.patients",
        "dataset.bboxes",
        "dataset.images",
        "dataset.findings",
        "dataset.image_labels",
        "dataset.versions",
        "dataset.splits",
        "dataset.quality_issues",
        "serving.jobs",
        "serving.predictions",
        "monitoring.drift_metrics",
    }


def test_fourteen_unique_classes():
    assert len(CLASSES) == 14
    assert len(set(CLASSES)) == 14
    assert NO_FINDING not in CLASSES


def test_migration_seeds_the_same_findings_in_the_same_order():
    migration = _load_initial_migration()

    assert migration.FINDINGS == CLASSES
    assert migration.SCHEMAS == SCHEMAS
