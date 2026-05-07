"""
LearnSpace — Ingest Router

Endpoints:
  POST /api/ingest/file          Upload a file (PDF, video, audio, notes)
  POST /api/ingest/url           Submit a YouTube or web URL
  GET  /api/ingest/status/{id}   Poll processing status
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from ..config import get_settings
from ..db import get_db
from ..ingestion import ingest_material

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# ── Allowed MIME types ─────────────────────────────────────────────────────────
ALLOWED = {
    # PDFs / ebooks
    "application/pdf": ("pdf", ".pdf"),
    "application/epub+zip": ("pdf", ".epub"),
    # Documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("notes", ".docx"),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ("notes", ".pptx"),
    "text/plain": ("notes", ".txt"),
    "text/markdown": ("notes", ".md"),
    # Video
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "video/x-matroska": ("video", ".mkv"),
    "video/webm": ("video", ".webm"),
    # Audio
    "audio/mpeg": ("audio", ".mp3"),
    "audio/mp4": ("audio", ".m4a"),
    "audio/wav": ("audio", ".wav"),
    "audio/x-wav": ("audio", ".wav"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class UrlIngestRequest(BaseModel):
    url: str
    name: Optional[str] = None      # override display name


class IngestResponse(BaseModel):
    material_id: str
    name: str
    type: str
    status: str
    message: str


class StatusResponse(BaseModel):
    material_id: str
    status: str
    chunk_count: int
    error_msg: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Upload a file. The file is saved to disk immediately; indexing
    happens in the background. Poll /status/{id} to track progress.
    """
    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {settings.max_upload_mb} MB")

    # Validate type
    content_type = file.content_type or ""
    if content_type not in ALLOWED:
        # Try guessing from extension
        ext = Path(file.filename or "").suffix.lower()
        mat_type = _type_from_ext(ext)
        if not mat_type:
            raise HTTPException(415, f"Unsupported file type: {content_type}")
    else:
        mat_type, _ = ALLOWED[content_type]

    # Save file
    material_id = uuid.uuid4().hex
    safe_name = _safe_filename(file.filename or f"upload_{material_id}")
    dest = os.path.join(settings.upload_dir, f"{material_id}_{safe_name}")

    with open(dest, "wb") as f:
        f.write(content)

    # Register in DB
    await db.execute(
        "INSERT INTO materials (id, name, type, source_path, status) VALUES (?, ?, ?, ?, 'pending')",
        (material_id, file.filename or safe_name, mat_type, dest),
    )
    await db.commit()

    # Kick off background ingestion
    background_tasks.add_task(_run_ingest, material_id, settings.sqlite_path)

    logger.info(f"File registered: {material_id} ({mat_type}): {file.filename}")
    return IngestResponse(
        material_id=material_id,
        name=file.filename or safe_name,
        type=mat_type,
        status="pending",
        message="File received. Indexing started in background.",
    )


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    request: UrlIngestRequest,
    background_tasks: BackgroundTasks,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Submit a YouTube or web URL for ingestion.
    """
    url = request.url.strip()
    is_youtube = any(x in url for x in ("youtube.com/watch", "youtu.be/"))
    mat_type = "youtube" if is_youtube else "link"

    material_id = uuid.uuid4().hex
    display_name = request.name or (f"YouTube: {url[:60]}" if is_youtube else url[:80])

    await db.execute(
        "INSERT INTO materials (id, name, type, source_path, status) VALUES (?, ?, ?, ?, 'pending')",
        (material_id, display_name, mat_type, url),
    )
    await db.commit()

    background_tasks.add_task(_run_ingest, material_id, settings.sqlite_path)

    logger.info(f"URL registered: {material_id} ({mat_type}): {url}")
    return IngestResponse(
        material_id=material_id,
        name=display_name,
        type=mat_type,
        status="pending",
        message="URL received. Transcript extraction started in background.",
    )


@router.get("/status/{material_id}", response_model=StatusResponse)
async def get_status(
    material_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "SELECT id, status, chunk_count, error_msg FROM materials WHERE id = ?",
        (material_id,),
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "Material not found")

    return StatusResponse(
        material_id=row["id"],
        status=row["status"],
        chunk_count=row["chunk_count"] or 0,
        error_msg=row["error_msg"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_ingest(material_id: str, db_path: str):
    """Background task wrapper — opens its own DB connection."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await ingest_material(material_id, db)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _type_from_ext(ext: str) -> Optional[str]:
    mapping = {
        ".pdf": "pdf", ".epub": "pdf",
        ".docx": "notes", ".pptx": "notes", ".txt": "notes", ".md": "notes",
        ".mp4": "video", ".mov": "video", ".mkv": "video", ".webm": "video",
        ".mp3": "audio", ".m4a": "audio", ".wav": "audio",
    }
    return mapping.get(ext)
