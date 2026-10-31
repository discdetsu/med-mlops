"""The gate flow writes `model_candidates` with SQL, but Django's migration owns the table
(ADR 0007). This checks the two agree on the real Postgres schema: the backend must have
started (`make up`), since its migrations create the table.

The end-to-end promotion check (approve → every worker serves vN → rollback) is manual; see
the README's "Promotion" section.
"""

import os
import uuid

import psycopg
import pytest

from cxr_flows.db import record_candidate

pytestmark = pytest.mark.integration


def _record(url: str, model: str, *, passed: bool, report: dict[str, object]) -> None:
    record_candidate(
        url,
        model_name=model,
        model_version="7",
        champion_version="6",
        gate_passed=passed,
        auto_promote=False,
        report=report,
        report_uri="s3://artifacts/models/x/7/gate/vs-champion-v6.html",
        mlflow_run_id="run",
        dataset_rev="rev",
    )


def test_gate_rows_fit_the_backend_table_and_regating_reopens_them() -> None:
    url = os.environ["APP_DATABASE_URL"]
    model = f"_it{uuid.uuid4().hex[:8]}"
    try:
        _record(url, model, passed=False, report={"passed": False})
        with psycopg.connect(url) as conn:
            # As if someone rejected it in admin.
            conn.execute(
                "UPDATE model_candidates SET status = 'rejected', decided_at = now() "
                "WHERE model_name = %s",
                (model,),
            )

        _record(url, model, passed=True, report={"passed": True, "macro_auroc": {"delta": 0.01}})

        with psycopg.connect(url) as conn:
            row = conn.execute(
                "SELECT status, gate_passed, auto_promote, decided_at, "
                "report -> 'macro_auroc' ->> 'delta', gated_at IS NOT NULL "
                "FROM model_candidates WHERE model_name = %s",
                (model,),
            ).fetchall()
        assert row == [("pending", True, False, None, "0.01", True)]
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM model_candidates WHERE model_name = %s", (model,))
