from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import settings

_pool: ConnectionPool | None = None

# Initialize connection
def init_pool() -> None:
    """Initialize the connection pool. Call once at startup."""
    global _pool

    _pool = ConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        open=True,
        kwargs={"row_factory": dict_row},
    )

# Close connection pool
def close_pool() -> None:
    """Close connection pool. Call at shutdown."""
    global _pool
    if _pool:
        _pool.close()
        _pool = None


# Get a connection from the pool
@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Get a database connection from the pool."""
    if not _pool:
        raise RuntimeError("Database pool not initialized")
    with _pool.connection as conn:
        yield conn