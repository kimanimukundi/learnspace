"""
LearnSpace — Flashcards Router

Endpoints:
  POST /api/flashcards/generate/{material_id}   Generate flashcards from material
  GET  /api/flashcards/{material_id}             List flashcards for a material
  POST /api/flashcards/{card_id}/review          Submit review result (spaced repetition)
  GET  /api/flashcards/due                       Get cards due for review today
  DELETE /api/flashcards/{card_id}               Delete a flashcard
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import get_db
from ..llm import generate_flashcards

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class Flashcard(BaseModel):
    id: str
    material_id: str
    question: str
    answer: str
    difficulty: int
    times_seen: int
    times_correct: int
    next_review: Optional[str] = None
    created_at: str


class GenerateRequest(BaseModel):
    count: int = 10      # number of flashcards to generate


class ReviewRequest(BaseModel):
    result: str          # "correct" | "incorrect" | "skip"


class ReviewResponse(BaseModel):
    card_id: str
    next_review: str
    interval_days: float
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/generate/{material_id}", response_model=List[Flashcard])
async def generate(
    material_id: str,
    request: GenerateRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Generate flashcards from a material's content using the LLM.
    Fetches the first ~4000 words of chunks as source text.
    """
    # Validate material exists and is ready
    async with db.execute(
        "SELECT status FROM materials WHERE id = ?", (material_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "Material not found")
    if row["status"] != "ready":
        raise HTTPException(409, f"Material not ready (status: {row['status']})")

    # Fetch source text
    async with db.execute(
        "SELECT text FROM chunks WHERE material_id = ? ORDER BY chunk_index LIMIT 15",
        (material_id,),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        raise HTTPException(422, "No content found for this material")

    text = "\n\n".join(r["text"] for r in rows)

    # LLM generation
    n = min(max(request.count, 1), 30)
    raw_json = await generate_flashcards(text, n=n)

    try:
        clean = raw_json.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        cards_data = json.loads(clean)
        if not isinstance(cards_data, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Flashcard JSON parse error: {e}\nRaw: {raw_json[:300]}")
        raise HTTPException(500, "Flashcard generation failed — invalid LLM response")

    # Persist to SQLite
    now = datetime.utcnow().isoformat()
    created = []
    for item in cards_data[:n]:
        card_id = uuid.uuid4().hex
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        difficulty = int(item.get("difficulty", 2))

        if not question or not answer:
            continue

        await db.execute(
            "INSERT INTO flashcards (id, material_id, question, answer, difficulty, next_review) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (card_id, material_id, question, answer, difficulty, now),
        )
        created.append(Flashcard(
            id=card_id,
            material_id=material_id,
            question=question,
            answer=answer,
            difficulty=difficulty,
            times_seen=0,
            times_correct=0,
            next_review=now,
            created_at=now,
        ))

    await db.commit()
    logger.info(f"Generated {len(created)} flashcards for material {material_id}")
    return created


@router.get("/due", response_model=List[Flashcard])
async def get_due_cards(
    material_id: Optional[str] = None,
    limit: int = 20,
    db: aiosqlite.Connection = Depends(get_db),
):
    """Return flashcards due for review (next_review <= now)."""
    now = datetime.utcnow().isoformat()

    if material_id:
        async with db.execute(
            "SELECT * FROM flashcards WHERE material_id = ? AND "
            "(next_review IS NULL OR next_review <= ?) "
            "ORDER BY next_review ASC LIMIT ?",
            (material_id, now, limit),
        ) as cur:
            rows = await cur.fetchall()
    else:
        async with db.execute(
            "SELECT * FROM flashcards WHERE next_review IS NULL OR next_review <= ? "
            "ORDER BY next_review ASC LIMIT ?",
            (now, limit),
        ) as cur:
            rows = await cur.fetchall()

    return [_row_to_card(r) for r in rows]


@router.get("/{material_id}", response_model=List[Flashcard])
async def list_cards(
    material_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "SELECT * FROM flashcards WHERE material_id = ? ORDER BY created_at ASC",
        (material_id,),
    ) as cur:
        rows = await cur.fetchall()

    return [_row_to_card(r) for r in rows]


@router.post("/{card_id}/review", response_model=ReviewResponse)
async def review_card(
    card_id: str,
    request: ReviewRequest,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Record a review result and compute the next review date using
    a simplified SM-2 spaced repetition algorithm.
    """
    async with db.execute(
        "SELECT * FROM flashcards WHERE id = ?", (card_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        raise HTTPException(404, "Flashcard not found")

    card = dict(row)
    correct = request.result == "correct"
    skip = request.result == "skip"

    # Update counts
    times_seen = card["times_seen"] + (0 if skip else 1)
    times_correct = card["times_correct"] + (1 if correct else 0)

    # SM-2 interval calculation
    interval_days = _sm2_interval(
        times_correct=times_correct,
        times_seen=times_seen,
        difficulty=card["difficulty"],
        correct=correct,
        skip=skip,
    )

    next_review = (datetime.utcnow() + timedelta(days=interval_days)).isoformat()

    await db.execute(
        "UPDATE flashcards SET times_seen=?, times_correct=?, next_review=? WHERE id=?",
        (times_seen, times_correct, next_review, card_id),
    )
    await db.commit()

    msg = f"Next review in {interval_days:.1f} day(s)."
    if correct:
        msg = f"✓ Correct! {msg}"
    elif skip:
        msg = f"⟶ Skipped. {msg}"
    else:
        msg = f"✗ Review again soon. {msg}"

    return ReviewResponse(
        card_id=card_id,
        next_review=next_review,
        interval_days=interval_days,
        message=msg,
    )


@router.delete("/{card_id}")
async def delete_card(
    card_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    await db.execute("DELETE FROM flashcards WHERE id = ?", (card_id,))
    await db.commit()
    return {"deleted": True, "card_id": card_id}


# ═══════════════════════════════════════════════════════════════════════════════
# SPACED REPETITION — simplified SM-2
# ═══════════════════════════════════════════════════════════════════════════════

def _sm2_interval(
    times_correct: int,
    times_seen: int,
    difficulty: int,
    correct: bool,
    skip: bool,
) -> float:
    if skip:
        return 1.0  # review tomorrow

    if not correct:
        return 0.25  # review in 6 hours

    # Base interval grows with consecutive correct answers
    # Difficulty multiplier: easy→faster growth, hard→slower
    base = [1, 3, 7, 14, 30, 60]
    diff_mult = {1: 1.5, 2: 1.0, 3: 0.6}.get(difficulty, 1.0)

    idx = min(times_correct - 1, len(base) - 1)
    interval = base[idx] * diff_mult
    return max(0.25, round(interval, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _row_to_card(row) -> Flashcard:
    return Flashcard(
        id=row["id"],
        material_id=row["material_id"],
        question=row["question"],
        answer=row["answer"],
        difficulty=row["difficulty"],
        times_seen=row["times_seen"],
        times_correct=row["times_correct"],
        next_review=row["next_review"],
        created_at=row["created_at"],
    )
