"""
LearnSpace — Materials Router

Endpoints:
  GET    /api/materials            List all materials
  GET    /api/materials/{id}       Get one material
  DELETE /api/materials/{id}       Delete material + its chunks
  GET    /api/materials/{id}/summary   AI-generated summary
  GET    /api/materials/{id}/chunks    Raw chunks (for debugging)
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import get_db
from ..llm import generate_summary
from ..vector_store import delete_material as vstore_delete

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class Material(BaseModel):
    id: str
    name: str
    type: str
    status: str
    chunk_count: int
    page_count: Optional[int] = None
    duration_s: Optional[int] = None
    error_msg: Optional[str] = None
    created_at: str


class SummarySection(BaseModel):
    heading: str
    content: str


class SummaryResponse(BaseModel):
    material_id: str
    title: str
    overview: str
    key_concepts: List[str]
    sections: List[SummarySection]
    study_tips: List[str]


class ChunkItem(BaseModel):
    id: str
    chunk_index: int
    text: str
    page_num: Optional[int] = None
    timestamp_s: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("", response_model=List[Material])
async def list_materials(
    status: Optional[str] = None,
    db: aiosqlite.Connection = Depends(get_db),
):
    if status:
        async with db.execute_query(
            "SELECT * FROM materials WHERE status = ? ORDER BY created_at DESC", (status,)
        ) as cur:
            rows = await cur.fetchall()
    else:
        async with db.execute_query(
            "SELECT * FROM materials ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()

    return [_row_to_material(r) for r in rows]


@router.get("/{material_id}", response_model=Material)
async def get_material(
    material_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute_query(
        "SELECT * FROM materials WHERE id = ?", (material_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "Material not found")

    return _row_to_material(row)


@router.delete("/{material_id}")
async def delete_material(
    material_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute_query("SELECT id FROM materials WHERE id = ?", (material_id,)) as cur:
        row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Material not found")

    # Remove from vector store
    await vstore_delete(material_id)

    # Remove from SQLite (chunks cascade)
    await db.execute("DELETE FROM materials WHERE id = ?", (material_id,))
    await db.commit()

    logger.info(f"Deleted material {material_id}")
    return {"deleted": True, "material_id": material_id}


@router.get("/{material_id}/summary", response_model=SummaryResponse)
async def get_summary(
    material_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Generate an AI summary from the material's stored chunks.
    The first ~6000 words of chunks are used as context.
    """
    async with db.execute_query(
        "SELECT status FROM materials WHERE id = ?", (material_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "Material not found")
    if row["status"] != "ready":
        raise HTTPException(409, f"Material not ready yet (status: {row['status']})")

    # Fetch enough chunks to fill ~6000 words
    async with db.execute_query(
        "SELECT text FROM chunks WHERE material_id = ? ORDER BY chunk_index LIMIT 20",
        (material_id,),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        raise HTTPException(422, "No content found for this material")

    combined = "\n\n".join(r["text"] for r in rows)
    raw_json = await generate_summary(combined)

    # Parse LLM JSON response
    try:
        # Strip accidental markdown fences
        clean = raw_json.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.error(f"LLM returned invalid JSON for summary: {raw_json[:200]}")
        raise HTTPException(500, "Summary generation failed — LLM returned invalid JSON")

    return SummaryResponse(
        material_id=material_id,
        title=data.get("title", "Summary"),
        overview=data.get("overview", ""),
        key_concepts=data.get("key_concepts", []),
        sections=[SummarySection(**s) for s in data.get("sections", [])],
        study_tips=data.get("study_tips", []),
    )


@router.get("/{material_id}/chunks", response_model=List[ChunkItem])
async def get_chunks(
    material_id: str,
    limit: int = 50,
    offset: int = 0,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute_query(
        "SELECT id, chunk_index, text, page_num, timestamp_s FROM chunks "
        "WHERE material_id = ? ORDER BY chunk_index LIMIT ? OFFSET ?",
        (material_id, limit, offset),
    ) as cur:
        rows = await cur.fetchall()

    return [
        ChunkItem(
            id=r["id"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            page_num=r["page_num"],
            timestamp_s=r["timestamp_s"],
        )
        for r in rows
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _row_to_material(row) -> Material:
    return Material(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        status=row["status"],
        chunk_count=row["chunk_count"] or 0,
        page_count=row["page_count"],
        duration_s=row["duration_s"],
        error_msg=row["error_msg"],
        created_at=row["created_at"],
    )
