"""
LearnSpace — Database Layer

Two stores:
  1. SQLite (via aiosqlite)  — material metadata, chat history, flashcard scores
  2. ChromaDB                — vector embeddings for semantic search
"""

import aiosqlite
import os
import logging
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── SQLite ─────────────────────────────────────────────────────────────────────

CREATE_MATERIALS = """
CREATE TABLE IF NOT EXISTS materials (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,         -- pdf | video | audio | notes | youtube | link
    source_path TEXT,                  -- local file path or original URL
    status      TEXT DEFAULT 'pending', -- pending | processing | ready | error
    page_count  INTEGER,
    duration_s  INTEGER,               -- for audio/video
    chunk_count INTEGER DEFAULT 0,
    error_msg   TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);
"""

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    material_id  TEXT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page_num     INTEGER,
    timestamp_s  REAL,                 -- for video/audio chunks
    created_at   TEXT DEFAULT (datetime('now'))
);
"""

CREATE_CHATS = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,        -- user | assistant
    content      TEXT NOT NULL,
    material_ids TEXT,                 -- JSON list of material IDs used as context
    created_at   TEXT DEFAULT (datetime('now'))
);
"""

CREATE_FLASHCARDS = """
CREATE TABLE IF NOT EXISTS flashcards (
    id           TEXT PRIMARY KEY,
    material_id  TEXT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    difficulty   INTEGER DEFAULT 2,    -- 1 easy  2 medium  3 hard
    times_seen   INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    next_review  TEXT,                 -- ISO datetime for spaced repetition
    created_at   TEXT DEFAULT (datetime('now'))
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_material ON chunks(material_id);",
    "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_fc_material ON flashcards(material_id);",
    "CREATE INDEX IF NOT EXISTS idx_fc_review ON flashcards(next_review);",
]


async def init_db():
    """Create all tables and directories on first run."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.chroma_path, exist_ok=True)

    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(CREATE_MATERIALS)
        await db.execute(CREATE_CHUNKS)
        await db.execute(CREATE_CHATS)
        await db.execute(CREATE_FLASHCARDS)
        for idx in CREATE_INDEXES:
            await db.execute(idx)
        await db.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        await db.commit()

    logger.info(f"SQLite initialised at {settings.sqlite_path}")


async def get_db():
    """Dependency — yields an open aiosqlite connection."""
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        yield db
