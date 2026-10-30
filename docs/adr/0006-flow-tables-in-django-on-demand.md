# 0006: Django reads the flows' tables only once a feature needs them, and app tables share one namespace

- Status: Accepted
- Date: 2026-10-30

## Context

ADR 0003 says two things about the dataset tables the flows own (`dataset_versions`, `dataset_images`, `images`, `labels`) that Phase 2 made untrue:

- "Django reads them through unmanaged models from Phase 2." The serving core doesn't read them. A prediction records `dataset_rev` as a string from the model's manifest, and nothing in the backend joins on it, so unmanaged models added now would be unused code.
- "Their names can't clash with Django's tables, which carry an app-label prefix." ADR 0005 gave the backend's tables explicit names (`inference_jobs`, `predictions`, `feedback`) so the monitor flow's SQL reads naturally, which drops that prefix.

## Decision

This replaces the paragraph of ADR 0003 that begins "Metadata lives in the app database too" where it covers Django; the rest of ADR 0003 stands.

- **Unmanaged models arrive with the first feature that reads the tables**, and cover only the tables that feature reads. The likely first one is Phase 5, where dataset versions and reviewer labels show up in Django admin. Until then the flows are the only code that touches these tables.
- **Flows and backend share one table namespace in the app database.** The flows own `dataset_versions`, `dataset_images`, `images` and `labels`; the backend owns `inference_jobs`, `predictions`, `feedback` and Django's own prefixed tables (`auth_*`, `django_*`). A new table on either side must not reuse a name the other side owns.

## Consequences

- Nothing enforces the shared namespace. A clash would not fail loudly: the flows create their tables with `CREATE TABLE IF NOT EXISTS`, so they would skip a table Django had already created under the same name and then fail on its columns. Reviewers have to check new table names against both lists.
- Until Phase 5, tracing a prediction back to its data means reading `dataset_versions` with SQL or the flows' code, not Django admin.

## Alternatives considered

- **Add the unmanaged models in Phase 2 as ADR 0003 planned:** four models and admin pages with no reader yet, which would have to track the flows' schema in the meantime.
- **Restore Django's app-label prefix on the backend's tables:** keeps ADR 0003's guarantee, but the monitor flow's SQL then reads `jobs_prediction` instead of `predictions`, which is the reason ADR 0005 chose explicit names.
