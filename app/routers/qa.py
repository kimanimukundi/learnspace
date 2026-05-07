"""
LearnSpace — Q&A Router

Endpoints:
  POST /api/qa/ask               Ask a question (RAG pipeline)
  GET  /api/qa/history/{session} Get chat history for a session
  DELETE /api/qa/history/{session} Clear a session's history
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import get_db
from ..llm import answer_question
from ..vector_store import search

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None        # omit to start a new session
    material_ids: Optional[List[str]] = None  # restrict context to these materials


class SourceChunk(BaseModel):
    material_id: str
    text: str
    score: float
    page_num: Optional[int] = None
    timestamp_s: Optional[float] = None


class AskResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk]


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: str


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Full RAG pipeline:
      1. Retrieve top-K relevant chunks from ChromaDB
      2. Load recent chat history from SQLite
      3. Call LLM (Claude or Ollama) with context + history
      4. Persist messages to SQLite
      5. Return answer + source citations
    """
    session_id = request.session_id or uuid.uuid4().hex
    question = request.question.strip()

    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # ── 1. Semantic retrieval ─────────────────────────────────────────────────
    hits = await search(
        query=question,
        material_ids=request.material_ids,
    )

    # ── 2. Load recent history (last 10 turns = 20 messages) ─────────────────
    history = await _load_history(db, session_id, limit=20)

    # ── 3. LLM call ───────────────────────────────────────────────────────────
    answer = await answer_question(
        question=question,
        context_chunks=hits,
        chat_history=history,
    )

    # ── 4. Persist user message + assistant reply ─────────────────────────────
    material_ids_used = list({h["material_id"] for h in hits})
    await _save_message(db, session_id, "user", question, material_ids_used)
    await _save_message(db, session_id, "assistant", answer, material_ids_used)

    # ── 5. Build source list ──────────────────────────────────────────────────
    sources = []
    seen = set()
    for hit in hits:
        key = (hit["material_id"], hit["text"][:60])
        if key in seen:
            continue
        seen.add(key)
        meta = hit.get("metadata", {})
        sources.append(SourceChunk(
            material_id=hit["material_id"],
            text=hit["text"][:300] + ("…" if len(hit["text"]) > 300 else ""),
            score=round(hit["score"], 3),
            page_num=meta.get("page_num") if (meta.get("page_num") or 0) > 0 else None,
            timestamp_s=meta.get("timestamp_s") if (meta.get("timestamp_s") or -1) >= 0 else None,
        ))

    return AskResponse(session_id=session_id, answer=answer, sources=sources)


@router.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_history(
    session_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "SELECT role, content, created_at FROM chat_messages "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()

    return [ChatMessage(role=r["role"], content=r["content"], created_at=r["created_at"]) for r in rows]


@router.delete("/history/{session_id}")
async def clear_history(
    session_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    await db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    await db.commit()
    return {"deleted": True, "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

async def _load_history(
    db: aiosqlite.Connection,
    session_id: str,
    limit: int = 20,
) -> List[dict]:
    async with db.execute(
        "SELECT role, content FROM chat_messages "
        "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()

    # Reverse so oldest first (for LLM context)
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def _save_message(
    db: aiosqlite.Connection,
    session_id: str,
    role: str,
    content: str,
    material_ids: List[str],
):
    msg_id = uuid.uuid4().hex
    await db.execute(
        "INSERT INTO chat_messages (id, session_id, role, content, material_ids) "
        "VALUES (?, ?, ?, ?, ?)",
        (msg_id, session_id, role, content, json.dumps(material_ids)),
    )
    await db.commit()
