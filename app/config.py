"""
LearnSpace — Configuration
All settings are read from environment variables (with sensible defaults).
Copy .env.example to .env and fill in your values.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_dir: str = os.path.join(base_dir, "uploads")
    data_dir: str = os.path.join(base_dir, "data")

    # ── Database ──────────────────────────────────────────────────────────────
    sqlite_path: str = os.path.join(base_dir, "data", "learnspace.db")
    chroma_path: str = os.path.join(base_dir, "data", "chroma")

   # ── AI — Online (Groq) ───────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # ── AI — Offline (Ollama) ─────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # "local"  → sentence-transformers (free, offline)
    # "openai" → text-embedding-3-small
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"    # ~80 MB, runs on CPU

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 500          # tokens per chunk
    chunk_overlap: int = 80        # overlap between chunks
    retrieval_top_k: int = 6       # number of chunks to retrieve per query

    # ── Whisper (transcription) ───────────────────────────────────────────────
    # "local"   → openai-whisper running on device
    # "api"     → OpenAI Whisper API  (requires OPENAI_API_KEY)
    whisper_provider: str = "local"
    whisper_model: str = "base"    # tiny / base / small / medium / large

    # ── YouTube ───────────────────────────────────────────────────────────────
    # Uses yt-dlp to download audio, then Whisper to transcribe.
    # Set to "transcript" to try the YouTube transcript API first (faster).
    youtube_strategy: str = "transcript"   # "transcript" | "whisper"

    # ── Misc ──────────────────────────────────────────────────────────────────
    max_upload_mb: int = 500
    prefer_offline: bool = False   # force offline mode even when online

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
