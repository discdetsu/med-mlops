from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from cxr.db import Base
from cxr.db.base import SCHEMAS

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url():
    # Tests pass the URL explicitly; otherwise read CXR_DB_* settings.
    if url := config.get_main_option("sqlalchemy.url"):
        return url
    from cxr.config import get_settings

    return get_settings().database_url


def _include_name(name, type_, parent_names):
    # Only manage our own schemas; ignore `public` and anything else in the database.
    if type_ == "schema":
        return name in SCHEMAS
    return True


_options = dict(
    target_metadata=target_metadata,
    include_schemas=True,
    include_name=_include_name,
    compare_server_default=True,
)


def run_migrations_offline() -> None:
    context.configure(
        url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"}, **_options
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, **_options)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
