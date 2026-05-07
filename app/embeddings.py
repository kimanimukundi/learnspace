"""
LearnSpace — Embedding Service

Supports:
  • local   → sentence-transformers (all-MiniLM-L6-v2, runs on CPU, ~80 MB)
  • openai  → text-embedding-3-small via OpenAI API

The local model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations
import logging
import asyncio
from functools import lru_cache
from typing import List

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lazy-load sentence-transformers so startup is fast ────────────────────────
@lru_cache(maxsize=1)
def _load_local_model():
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {settings.embedding_model}")
        model = SentenceTransformer(settings.embedding_model)
        logger.info("Local embedding model loaded.")
        return model
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings.
    Returns a list of float vectors (one per input text).
    Runs CPU-bound work in a thread pool to avoid blocking the event loop.
    """
    if not texts:
        return []

    provider = settings.embedding_provider

    if provider == "local":
        loop = asyncio.get_event_loop()
        model = _load_local_model()
        embeddings = await loop.run_in_executor(
            None, lambda: model.encode(texts, show_progress_bar=False).tolist()
        )
        return embeddings

    elif provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        client = AsyncOpenAI()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


async def embed_single(text: str) -> List[float]:
    """Convenience wrapper for a single string."""
    results = await embed_texts([text])
    return results[0]
