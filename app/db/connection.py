from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

_pool: AsyncConnectionPool | None = None

# Initialize connection
async def init_pool() -> None:
    """Initialize the connection pool. Call once at startup."""
    global _pool

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        open=False,
        kwargs={"row_factory": dict_row},
    )
    await _pool.open()

# Close connection pool
async def close_pool() -> None:
    """Close connection pool. Call at shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# Get a connection from the pool
@asynccontextmanager
async def get_conn() -> AsyncIterator[psycopg.AsyncConnection]:
    """Get a database connection from the pool."""
    if not _pool:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    async with _pool.connection() as conn:
        yield conn