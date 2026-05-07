"""
LearnSpace — Ingestion Pipeline

Handles extraction and chunking for every supported content type:
  • PDF           → PyMuPDF (fitz)
  • DOCX / PPTX   → python-docx / python-pptx
  • TXT / MD      → plain read
  • Video / Audio → Whisper (local or API)
  • YouTube URL   → youtube-transcript-api first, Whisper fallback
  • Web link      → trafilatura (clean article extraction)

After extraction, text is split into overlapping chunks and stored in
ChromaDB + SQLite.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

import aiosqlite

from .config import get_settings
from .vector_store import add_chunks

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    Returns a list of chunk strings.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTORS — one per content type
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_pdf(path: str) -> Tuple[List[str], int]:
    """
    Extract text from a PDF, page by page.
    Returns (list_of_page_texts, page_count).
    """
    loop = asyncio.get_event_loop()

    def _extract():
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

        doc = fitz.open(path)
        pages = []
        page_count = doc.page_count   # ← read BEFORE closing
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return pages, page_count      # ← now safe to return

    return await loop.run_in_executor(None, _extract)


async def extract_docx(path: str) -> str:
    loop = asyncio.get_event_loop()

    def _extract():
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    return await loop.run_in_executor(None, _extract)


async def extract_pptx(path: str) -> str:
    loop = asyncio.get_event_loop()

    def _extract():
        try:
            from pptx import Presentation
        except ImportError:
            raise RuntimeError("python-pptx not installed. Run: pip install python-pptx")
        prs = Presentation(path)
        parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
        return "\n".join(parts)

    return await loop.run_in_executor(None, _extract)


async def extract_plain(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


async def transcribe_audio(path: str) -> str:
    """
    Transcribe audio/video using Groq's Whisper API.
    Extracts and compresses audio with ffmpeg first to stay under 25MB limit.
    Runs in a thread executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    def _run():
        import subprocess
        import tempfile

        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("groq not installed. Run: pip install groq")

        # Extract and compress audio using ffmpeg
        tmp_audio = path + "_compressed.mp3"
        try:
            result = subprocess.run([
                "ffmpeg", "-i", path,
                "-vn",          # strip video, audio only
                "-ar", "16000", # 16kHz sample rate (sufficient for speech)
                "-ac", "1",     # mono channel
                "-b:a", "32k",  # 32kbps bitrate — keeps file tiny
                "-y",           # overwrite if exists
                tmp_audio
            ], check=True, capture_output=True)

            # Check compressed size
            size_mb = os.path.getsize(tmp_audio) / (1024 * 1024)
            if size_mb > 24:
                raise ValueError(
                    f"Audio is still {size_mb:.1f}MB after compression. "
                    f"This video is too long for transcription. "
                    f"Please upload to YouTube and paste the link instead."
                )

            client = Groq(api_key=settings.groq_api_key)
            with open(tmp_audio, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_audio), f.read()),
                    model="whisper-large-v3-turbo",
                )
            return transcription.text

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed to process the video. "
                f"Make sure ffmpeg is installed and the file is not corrupted.\n"
                f"Details: {e.stderr.decode() if e.stderr else str(e)}"
            )
        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)

    return await loop.run_in_executor(None, _run)


async def _transcribe_local(path: str) -> str:
    loop = asyncio.get_event_loop()

    def _run():
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
        logger.info(f"Transcribing {path} with Whisper ({settings.whisper_model})…")
        model = whisper.load_model(settings.whisper_model)
        result = model.transcribe(path)
        return result["text"]

    return await loop.run_in_executor(None, _run)


async def _transcribe_openai_api(path: str) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai not installed. Run: pip install openai")

    client = AsyncOpenAI()
    with open(path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    return transcript.text


async def extract_youtube(url: str) -> Tuple[str, Optional[str]]:
    """
    Fetch YouTube transcript using youtube-transcript-api.
    Fast — no audio download needed.
    """
    try:
        text, title = await _youtube_transcript_api(url)
        return text, title
    except Exception as e:
        raise ValueError(
            f"Could not get transcript for this video: {e}\n"
            "Make sure the video has captions/subtitles enabled."
        )


async def _youtube_transcript_api(url: str) -> Tuple[str, str]:
    loop = asyncio.get_event_loop()

    def _run():
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise RuntimeError("youtube-transcript-api not installed.")

        import re as _re
        match = _re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            raise ValueError(f"Could not parse video ID from: {url}")
        video_id = match.group(1)

        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched)
        return text, f"YouTube ({video_id})"

    return await loop.run_in_executor(None, _run)


async def _youtube_whisper(url: str) -> Tuple[str, str]:
    loop = asyncio.get_event_loop()
    tmp_path = os.path.join(settings.upload_dir, f"_yt_{uuid.uuid4().hex}.mp3")

    def _download():
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": tmp_path.replace(".mp3", ""),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info.get("title", url)

    title = await loop.run_in_executor(None, _download)
    text = await transcribe_audio(tmp_path)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return text, title


async def extract_web(url: str) -> Tuple[str, str]:
    """
    Extract clean article text from a web URL using trafilatura.
    Returns (text, title).
    """
    loop = asyncio.get_event_loop()

    def _run():
        try:
            import trafilatura
        except ImportError:
            raise RuntimeError(
                "trafilatura not installed. Run: pip install trafilatura"
            )
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(f"Could not fetch URL: {url}")
        text = trafilatura.extract(downloaded) or ""
        metadata = trafilatura.extract_metadata(downloaded)
        title = (metadata.title if metadata else None) or url
        return text, title

    return await loop.run_in_executor(None, _run)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INGESTION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

async def ingest_material(material_id: str, db: aiosqlite.Connection):
    """
    Full ingestion pipeline for one material.
    Called as a background task after the file/URL is registered.

    Flow:
      1. Load material record from SQLite
      2. Extract raw text (dispatcher based on type)
      3. Chunk text
      4. Store chunks in SQLite + embeddings in ChromaDB
      5. Update material status → 'ready'
    """
    async with db.execute(
        "SELECT * FROM materials WHERE id = ?", (material_id,)
    ) as cur:
        row = await cur.fetchone()

    if not row:
        logger.error(f"Material {material_id} not found in DB")
        return

    mat = dict(row)
    logger.info(f"Ingesting material {material_id} ({mat['type']}): {mat['name']}")

    await _update_status(db, material_id, "processing")

    try:
        chunks_with_meta = await _extract_and_chunk(mat)
        await _store_chunks(db, material_id, chunks_with_meta)
        await _update_status(
            db, material_id, "ready",
            chunk_count=len(chunks_with_meta)
        )
        logger.info(f"Material {material_id} ready — {len(chunks_with_meta)} chunks")

    except Exception as exc:
        logger.exception(f"Ingestion failed for {material_id}")
        await _update_status(db, material_id, "error", error_msg=str(exc))


async def _extract_and_chunk(mat: dict) -> List[dict]:
    """
    Dispatcher — extract text, split into chunks.
    Returns list of dicts: {text, chunk_index, page_num?, timestamp_s?}
    """
    mat_type = mat["type"]
    source = mat["source_path"]
    chunks_meta = []

    if mat_type == "pdf":
        pages, _ = await extract_pdf(source)
        for page_num, page_text in enumerate(pages, 1):
            for chunk in chunk_text(page_text):
                chunks_meta.append({
                    "text": chunk,
                    "page_num": page_num,
                    "timestamp_s": None,
                })

    elif mat_type == "notes":
        ext = Path(source).suffix.lower()
        if ext == ".docx":
            raw = await extract_docx(source)
        elif ext == ".pptx":
            raw = await extract_pptx(source)
        else:
            raw = await extract_plain(source)
        for chunk in chunk_text(raw):
            chunks_meta.append({"text": chunk, "page_num": None, "timestamp_s": None})

    elif mat_type in ("video", "audio"):
        raw = await transcribe_audio(source)
        # For timed chunks we could use Whisper's word timestamps —
        # simplified here to plain chunks.
        for chunk in chunk_text(raw):
            chunks_meta.append({"text": chunk, "page_num": None, "timestamp_s": None})

    elif mat_type == "youtube":
        raw, _ = await extract_youtube(source)
        for chunk in chunk_text(raw):
            chunks_meta.append({"text": chunk, "page_num": None, "timestamp_s": None})

    elif mat_type == "link":
        raw, _ = await extract_web(source)
        for chunk in chunk_text(raw):
            chunks_meta.append({"text": chunk, "page_num": None, "timestamp_s": None})

    else:
        raise ValueError(f"Unknown material type: {mat_type}")

    # Assign sequential chunk indices
    for i, item in enumerate(chunks_meta):
        item["chunk_index"] = i

    return chunks_meta


async def _store_chunks(
    db: aiosqlite.Connection,
    material_id: str,
    chunks_meta: List[dict],
):
    chunk_ids = []
    texts = []
    chroma_meta = []

    rows = []
    for item in chunks_meta:
        cid = f"{material_id}_chunk_{item['chunk_index']}"
        chunk_ids.append(cid)
        texts.append(item["text"])
        chroma_meta.append({
            "material_id": material_id,
            "chunk_index": item["chunk_index"],
            "page_num": item.get("page_num") or -1,
            "timestamp_s": item.get("timestamp_s") or -1.0,
        })
        rows.append((
            cid, material_id,
            item["chunk_index"], item["text"],
            item.get("page_num"), item.get("timestamp_s"),
        ))

    # SQLite
    await db.executemany(
        "INSERT OR REPLACE INTO chunks "
        "(id, material_id, chunk_index, text, page_num, timestamp_s) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    await db.commit()

    # ChromaDB (with embeddings)
    await add_chunks(material_id, chunk_ids, texts, chroma_meta)


async def _update_status(
    db: aiosqlite.Connection,
    material_id: str,
    status: str,
    chunk_count: int = None,
    error_msg: str = None,
):
    if chunk_count is not None:
        await db.execute(
            "UPDATE materials SET status=?, chunk_count=?, updated_at=datetime('now') "
            "WHERE id=?",
            (status, chunk_count, material_id),
        )
    elif error_msg is not None:
        await db.execute(
            "UPDATE materials SET status=?, error_msg=?, updated_at=datetime('now') "
            "WHERE id=?",
            (status, error_msg, material_id),
        )
    else:
        await db.execute(
            "UPDATE materials SET status=?, updated_at=datetime('now') WHERE id=?",
            (status, material_id),
        )
    await db.commit()
