#!/usr/bin/env bash
# Runs once, on first start with an empty data volume.
# One Postgres instance, one database + owner role per service.
set -euo pipefail

create_db() {
    local name="$1" password="$2"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -v name="$name" -v password="$password" <<'EOSQL'
CREATE ROLE :"name" LOGIN PASSWORD :'password';
CREATE DATABASE :"name" OWNER :"name";
REVOKE ALL ON DATABASE :"name" FROM PUBLIC;
EOSQL
}

create_db app "$APP_DB_PASSWORD"
create_db mlflow "$MLFLOW_DB_PASSWORD"
create_db airflow "$AIRFLOW_DB_PASSWORD"
