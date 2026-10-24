#!/bin/sh
# Runs once on an empty data volume. One database per service, each owned by its
# own user, with PUBLIC connect revoked so services can't read each other's data.
set -eu

create_db() {
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -v db="$1" -v usr="$1" -v pw="$2" <<'SQL'
CREATE USER :"usr" WITH PASSWORD :'pw';
CREATE DATABASE :"db" OWNER :"usr";
REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
SQL
}

create_db app "$APP_DB_PASSWORD"
create_db mlflow "$MLFLOW_DB_PASSWORD"
create_db prefect "$PREFECT_DB_PASSWORD"
create_db labelstudio "$LABELSTUDIO_DB_PASSWORD"
