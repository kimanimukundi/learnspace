"""
LearnSpace — LLM Service

Unified interface for:
  • Online  → Anthropic Claude (claude-sonnet-4-5 / claude-haiku-4-5)
  • Offline → Ollama local models (llama3.2, mistral, etc.)

Auto-detects which to use based on settings.prefer_offline and
whether the Anthropic API key is set.
"""

from __future__ import annotations

import logging
from typing import List, AsyncGenerator

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── System prompts ─────────────────────────────────────────────────────────────

QA_SYSTEM = """You are LearnSpace, a personal AI tutor. You help users understand \
their uploaded study materials — PDFs, lecture transcripts, notes, and more.

Rules:
- ONLY answer based on the provided context chunks. Do not hallucinate.
- If the answer is not in the context, say so honestly.
- Be clear, educational, and encouraging.
- When referencing specific material, mention the source name and location \
  (page number or timestamp) if available.
- Use markdown for clarity: headers, bullet points, bold for key terms.
- Keep answers concise but complete. Aim for 150–300 words unless the question \
  demands more depth.
"""

FLASHCARD_SYSTEM = """You are an expert educator creating study flashcards.
Generate exactly {n} high-quality question-answer pairs from the provided text.

Requirements:
- Cover the most important concepts, definitions, and facts.
- Questions should test genuine understanding, not just recall.
- Answers should be concise (1–4 sentences).
- Vary difficulty: some conceptual, some definitional, some applied.

Respond ONLY with a JSON array, no markdown fences:
[
  {{"question": "...", "answer": "...", "difficulty": 1}},
  ...
]
difficulty: 1=easy, 2=medium, 3=hard
"""

SUMMARY_SYSTEM = """You are an expert at summarising study materials.
Create a structured summary of the provided text.

Format your response as JSON (no markdown fences):
{{
  "title": "...",
  "overview": "2-3 sentence overview",
  "key_concepts": ["concept 1", "concept 2", ...],
  "sections": [
    {{"heading": "...", "content": "..."}},
    ...
  ],
  "study_tips": ["tip 1", "tip 2"]
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def answer_question(
    question: str,
    context_chunks: List[dict],
    chat_history: List[dict] = None,
) -> str:
    """
    Generate an answer grounded in the retrieved context chunks.

    Args:
        question:      The user's question
        context_chunks: List of {text, material_id, metadata} dicts from vector search
        chat_history:  Previous messages [{role, content}]

    Returns:
        Assistant reply as a string (may contain markdown)
    """
    context_str = _format_context(context_chunks)
    history = chat_history or []

    messages = [
        *history,
        {
            "role": "user",
            "content": (
                f"<context>\n{context_str}\n</context>\n\n"
                f"Question: {question}"
            ),
        },
    ]

    return await _complete(QA_SYSTEM, messages)


async def generate_flashcards(text: str, n: int = 10) -> str:
    """
    Generate n flashcards from text. Returns raw JSON string.
    """
    messages = [{"role": "user", "content": f"Text to create flashcards from:\n\n{text}"}]
    system = FLASHCARD_SYSTEM.format(n=n)
    return await _complete(system, messages, max_tokens=2000)


async def generate_summary(text: str) -> str:
    """
    Summarise text. Returns raw JSON string.
    """
    messages = [{"role": "user", "content": f"Summarise this study material:\n\n{text}"}]
    return await _complete(SUMMARY_SYSTEM, messages, max_tokens=1500)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _should_use_offline() -> bool:
    if settings.prefer_offline:
        return True
    if not settings.groq_api_key:
        return True
    return False

def _format_context(chunks: List[dict]) -> str:
    if not chunks:
        return "(No relevant context found in your materials.)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        location = ""
        if meta.get("page_num") and meta["page_num"] > 0:
            location = f" [page {meta['page_num']}]"
        elif meta.get("timestamp_s") and meta["timestamp_s"] >= 0:
            ts = int(meta["timestamp_s"])
            location = f" [{ts // 60}:{ts % 60:02d}]"

        mat_id = chunk.get("material_id", "unknown")
        parts.append(f"[{i}] Source: {mat_id}{location}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


async def _complete(system, messages, max_tokens=1000):
    if _should_use_offline():
        return await _ollama_complete(system, messages, max_tokens)
    return await _groq_complete(system, messages, max_tokens)

async def _groq_complete(
    system: str,
    messages: List[dict],
    max_tokens: int,
) -> str:
    try:
        from groq import AsyncGroq
    except ImportError:
        raise RuntimeError("groq not installed. Run: pip install groq")

    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=settings.groq_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            *messages,
        ],
    )
    return response.choices[0].message.content


async def _ollama_complete(
    system: str,
    messages: List[dict],
    max_tokens: int,
) -> str:
    """
    Call a locally running Ollama model.
    Ollama must be installed and running: https://ollama.ai
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed. Run: pip install httpx")

    # Ollama chat API
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except httpx.ConnectError:
            return (
                "⚠️ Offline mode: Could not connect to Ollama. "
                "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {settings.ollama_model}`)."
            )
