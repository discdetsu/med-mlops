"""Settings loaded from environment variables (and `.env` when present).

Every service reads its config through `get_settings()`. Variables are prefixed
with `CXR_`; see `.env.example` for the full list.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CXR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres `app` database (the `mlflow` and `airflow` databases are owned by those services)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "app"
    db_user: str = "app"
    db_password: SecretStr

    # RustFS (S3-compatible)
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str
    s3_secret_key: SecretStr
    s3_bucket: str = "cxr"

    mlflow_tracking_uri: str = "http://localhost:5000"

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.db_user,
            password=self.db_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
