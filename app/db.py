"""
LearnSpace — Database Layer

Supports both SQLite (local dev) and PostgreSQL (production).
Automatically detects which to use based on DATABASE_URL env var.
"""

import logging
import os
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Detect which DB to use ─────────────────────────────────────────────────────
def _is_postgres() -> bool:
    return bool(settings.database_url and settings.database_url.startswith("postgres"))

def get_db_url() -> str:
    if _is_postgres():
        return settings.database_url
    return f"sqlite+aiosqlite:///{settings.sqlite_path}"


# ── SQL — works for both SQLite and PostgreSQL ─────────────────────────────────

CREATE_MATERIALS = """
CREATE TABLE IF NOT EXISTS materials (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    source_path TEXT,
    status      TEXT DEFAULT 'pending',
    page_count  INTEGER,
    duration_s  INTEGER,
    chunk_count INTEGER DEFAULT 0,
    error_msg   TEXT,
    created_at  TEXT DEFAULT (CURRENT_TIMESTAMP),
    updated_at  TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    material_id  TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page_num     INTEGER,
    timestamp_s  REAL,
    created_at   TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""

CREATE_CHATS = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    material_ids TEXT,
    created_at   TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""

CREATE_FLASHCARDS = """
CREATE TABLE IF NOT EXISTS flashcards (
    id            TEXT PRIMARY KEY,
    material_id   TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    difficulty    INTEGER DEFAULT 2,
    times_seen    INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    next_review   TEXT,
    created_at    TEXT DEFAULT (CURRENT_TIMESTAMP)
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_material ON chunks(material_id);",
    "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_fc_material ON flashcards(material_id);",
    "CREATE INDEX IF NOT EXISTS idx_fc_review ON flashcards(next_review);",
]


async def init_db():
    """Create all tables on first run."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.chroma_path, exist_ok=True)

    if _is_postgres():
        await _init_postgres()
    else:
        await _init_sqlite()


async def _init_postgres():
    import asyncpg
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(CREATE_MATERIALS)
        await conn.execute(CREATE_CHUNKS)
        await conn.execute(CREATE_CHATS)
        await conn.execute(CREATE_FLASHCARDS)
        for idx in CREATE_INDEXES:
            try:
                await conn.execute(idx)
            except Exception:
                pass  # index may already exist
        logger.info("PostgreSQL initialised")
    finally:
        await conn.close()


async def _init_sqlite():
    import aiosqlite
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(CREATE_MATERIALS)
        await db.execute(CREATE_CHUNKS)
        await db.execute(CREATE_CHATS)
        await db.execute(CREATE_FLASHCARDS)
        for idx in CREATE_INDEXES:
            await db.execute(idx)
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.commit()
    logger.info(f"SQLite initialised at {settings.sqlite_path}")


async def get_db():
    """
    Dependency — yields a connection that works like aiosqlite.
    For PostgreSQL, wraps asyncpg in a compatibility shim.
    For SQLite, returns aiosqlite connection as before.
    """
    if _is_postgres():
        async for conn in _get_postgres_db():
            yield conn
    else:
        async for conn in _get_sqlite_db():
            yield conn


async def _get_sqlite_db():
    import aiosqlite
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def _get_postgres_db():
    import asyncpg
    conn = await asyncpg.connect(settings.database_url)
    try:
        yield PostgresCompat(conn)
    finally:
        await conn.close()


class PostgresCompat:
    """
    Thin compatibility shim that makes asyncpg behave like aiosqlite,
    so all routers work without changes.
    """

    def __init__(self, conn):
        self._conn = conn
        self._tx = None

    async def execute(self, sql: str, params: tuple = ()):
        # Convert SQLite ? placeholders to PostgreSQL $1, $2...
        sql = _convert_placeholders(sql)
        await self._conn.execute(sql, *params)

    async def executemany(self, sql: str, params_list):
        sql = _convert_placeholders(sql)
        await self._conn.executemany(sql, params_list)

    async def commit(self):
        pass  # asyncpg auto-commits by default

    def execute_query(self, sql: str, params: tuple = ()):
        sql = _convert_placeholders(sql)
        return _AsyncCursor(self._conn, sql, params)


class _AsyncCursor:
    def __init__(self, conn, sql, params):
        self._conn = conn
        self._sql = sql
        self._params = params
        self._rows = None

    async def __aenter__(self):
        self._rows = await self._conn.fetch(self._sql, *self._params)
        return self

    async def __aexit__(self, *args):
        pass

    async def fetchone(self):
        if not self._rows:
            return None
        return PostgresRow(self._rows[0])

    async def fetchall(self):
        return [PostgresRow(r) for r in self._rows]


class PostgresRow:
    """Makes asyncpg Record behave like aiosqlite.Row (dict-like access)."""
    def __init__(self, record):
        self._record = record

    def __getitem__(self, key):
        return self._record[key]

    def keys(self):
        return self._record.keys()

    def __iter__(self):
        return iter(self._record.keys())


def _convert_placeholders(sql: str) -> str:
    """Convert SQLite ? to PostgreSQL $1, $2, $3..."""
    import re
    count = 0
    def replacer(match):
        nonlocal count
        count += 1
        return f'${count}'
    return re.sub(r'\?', replacer, sql)