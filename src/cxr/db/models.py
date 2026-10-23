"""ORM models for the `app` database, following `ERD.png`.

Three schemas:
- `dataset`: patients, images, findings, bounding boxes, versioned patient-level splits,
  ETL runs and data quality issues
- `serving`: inference jobs and their per-finding scores
- `monitoring`: drift results written by `drift_job`

Status-like columns use text + CHECK instead of Postgres enums, so adding a value
is a one-line migration.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from cxr.db.base import Base


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# --- dataset -------------------------------------------------------------------------


class EtlRun(Base):
    __tablename__ = "etl_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'success', 'failed')", name="status"),
        {"schema": "dataset"},
    )

    etl_run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    started_at: Mapped[datetime] = _created_at()
    status: Mapped[str] = mapped_column(Text, server_default="running")
    rows_loaded: Mapped[int | None] = mapped_column(Integer)
    # sha256 of the source metadata CSV, recorded at extract time.
    source_sha: Mapped[str | None] = mapped_column(String(64))


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F')", name="gender"),
        {"schema": "dataset"},
    )

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    gender: Mapped[str] = mapped_column(String(1))


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        CheckConstraint("age_years BETWEEN 0 AND 120", name="age_years"),
        CheckConstraint("view_position IN ('PA', 'AP')", name="view_position"),
        CheckConstraint("char_length(sha256) = 64", name="sha256"),
        {"schema": "dataset"},
    )

    # The NIH file name, e.g. `00000001_000.png`.
    image_id: Mapped[str] = mapped_column(Text, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("dataset.patients.patient_id"), index=True)
    follow_up_no: Mapped[int] = mapped_column(Integer)
    age_years: Mapped[int] = mapped_column(SmallInteger)
    view_position: Mapped[str] = mapped_column(String(2))
    storage_uri: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))


class Finding(Base):
    """The 14 findings; `finding_id` is the model output index (see `cxr.labels.CLASSES`)."""

    __tablename__ = "findings"
    __table_args__ = ({"schema": "dataset"},)

    finding_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(Text, unique=True)


class ImageLabel(Base):
    """One row per positive finding. An image with no rows is `No Finding`."""

    __tablename__ = "image_labels"
    __table_args__ = ({"schema": "dataset"},)

    image_id: Mapped[str] = mapped_column(
        ForeignKey("dataset.images.image_id", ondelete="CASCADE"), primary_key=True
    )
    finding_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.findings.finding_id"), primary_key=True
    )


class BBox(Base):
    """Expert bounding boxes from `BBox_List_2017.csv`, in original image pixels."""

    __tablename__ = "bboxes"
    __table_args__ = (
        CheckConstraint("w > 0 AND h > 0", name="positive_size"),
        {"schema": "dataset"},
    )

    bbox_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    image_id: Mapped[str] = mapped_column(
        ForeignKey("dataset.images.image_id", ondelete="CASCADE"), index=True
    )
    finding_id: Mapped[int] = mapped_column(ForeignKey("dataset.findings.finding_id"))
    x: Mapped[float] = mapped_column(Double)
    y: Mapped[float] = mapped_column(Double)
    w: Mapped[float] = mapped_column(Double)
    h: Mapped[float] = mapped_column(Double)


class Version(Base):
    """A frozen dataset version (e.g. `cxr14-v1`) and its published manifest."""

    __tablename__ = "versions"
    __table_args__ = (
        CheckConstraint(
            "manifest_sha256 IS NULL OR char_length(manifest_sha256) = 64", name="manifest_sha256"
        ),
        {"schema": "dataset"},
    )

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created_at()


class Split(Base):
    """Keyed by patient, so no patient can be in two splits of the same version."""

    __tablename__ = "splits"
    __table_args__ = (
        CheckConstraint("split IN ('train', 'val', 'test')", name="split"),
        {"schema": "dataset"},
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.versions.version_id", ondelete="CASCADE"), primary_key=True
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.patients.patient_id"), primary_key=True
    )
    split: Mapped[str] = mapped_column(Text)


class QualityIssue(Base):
    """A rejected or suspicious source row/file. One row per (image, issue)."""

    __tablename__ = "quality_issues"
    __table_args__ = ({"schema": "dataset"},)

    # Not a foreign key: the image may be missing or rejected.
    image_id: Mapped[str] = mapped_column(Text, primary_key=True)
    issue: Mapped[str] = mapped_column(Text, primary_key=True)
    etl_run_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.etl_runs.etl_run_id", ondelete="CASCADE"), index=True
    )
    raw_value: Mapped[str | None] = mapped_column(Text)


# --- serving -------------------------------------------------------------------------


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'done', 'failed')", name="status"),
        {"schema": "serving"},
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    image_uri: Mapped[str] = mapped_column(Text)
    heatmap_uri: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("score BETWEEN 0 AND 1", name="score"),
        {"schema": "serving"},
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("serving.jobs.job_id", ondelete="CASCADE"), primary_key=True
    )
    finding_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.findings.finding_id"), primary_key=True
    )
    # Calibrated probability (after temperature scaling).
    score: Mapped[float] = mapped_column(Double)


# --- monitoring ----------------------------------------------------------------------


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    __table_args__ = (
        CheckConstraint("signal IN ('embedding', 'score', 'image_stats')", name="signal"),
        {"schema": "monitoring"},
    )

    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    signal: Mapped[str] = mapped_column(Text, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str | None] = mapped_column(Text)
    drift_score: Mapped[float] = mapped_column(Double)
    drift_detected: Mapped[bool]
