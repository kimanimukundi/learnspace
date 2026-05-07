"""
LearnSpace — Vector Store (ChromaDB)

Handles:
  • Adding chunks with their embeddings
  • Semantic similarity search
  • Deleting a material's chunks

ChromaDB persists to disk at settings.chroma_path — fully offline.
"""

from __future__ import annotations
import logging
from typing import List, Optional
from functools import lru_cache

from .config import get_settings
from .embeddings import embed_texts, embed_single

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION_NAME = "learnspace_chunks"


@lru_cache(maxsize=1)
def _get_client():
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    client = chromadb.PersistentClient(path=settings.chroma_path)
    logger.info(f"ChromaDB initialised at {settings.chroma_path}")
    return client


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


async def add_chunks(
    material_id: str,
    chunk_ids: List[str],
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
):
    """
    Embed and store a batch of chunks for a material.

    Args:
        material_id: ID of the parent material
        chunk_ids:   Unique IDs for each chunk (e.g. "mat123_chunk_0")
        texts:       Raw text for each chunk
        metadatas:   Optional list of dicts (page_num, timestamp_s, etc.)
    """
    if not texts:
        return

    if metadatas is None:
        metadatas = [{"material_id": material_id} for _ in texts]
    else:
        for m in metadatas:
            m["material_id"] = material_id

    # Embed in batches of 64 to avoid OOM on large documents
    BATCH = 64
    collection = _get_collection()

    for start in range(0, len(texts), BATCH):
        batch_texts = texts[start : start + BATCH]
        batch_ids = chunk_ids[start : start + BATCH]
        batch_meta = metadatas[start : start + BATCH]

        embeddings = await embed_texts(batch_texts)
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    logger.info(f"Stored {len(texts)} chunks for material {material_id}")


async def search(
    query: str,
    material_ids: Optional[List[str]] = None,
    top_k: int = None,
) -> List[dict]:
    """
    Semantic search over stored chunks.

    Args:
        query:        The user's question
        material_ids: If provided, restrict search to these materials
        top_k:        Number of results to return

    Returns:
        List of dicts with keys: id, text, material_id, score, metadata
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    query_embedding = await embed_single(query)
    collection = _get_collection()

    where = None
    if material_ids and len(material_ids) == 1:
        where = {"material_id": material_ids[0]}
    elif material_ids:
        where = {"material_id": {"$in": material_ids}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "material_id": results["metadatas"][0][i].get("material_id", ""),
            "score": 1 - results["distances"][0][i],  # cosine → similarity
            "metadata": results["metadatas"][0][i],
        })

    return hits


async def delete_material(material_id: str):
    """Remove all chunks belonging to a material."""
    collection = _get_collection()
    collection.delete(where={"material_id": material_id})
    logger.info(f"Deleted all chunks for material {material_id}")
