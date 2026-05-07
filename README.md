# LearnSpace — Backend

Personal AI learning assistant. Upload PDFs, videos, lecture notes, YouTube links, or web articles and chat with your materials — including fully offline.

---

## Architecture

```
learnspace/
├── app/
│   ├── main.py          ← FastAPI app + CORS + router registration
│   ├── config.py        ← All settings (env vars with defaults)
│   ├── db.py            ← SQLite schema + ChromaDB init
│   ├── ingestion.py     ← Extract → chunk → embed pipeline
│   ├── embeddings.py    ← Local (sentence-transformers) or OpenAI embeddings
│   ├── vector_store.py  ← ChromaDB wrapper (add, search, delete)
│   ├── llm.py           ← Claude (online) + Ollama (offline) LLM calls
│   └── routers/
│       ├── ingest.py    ← POST /api/ingest/file  POST /api/ingest/url
│       ├── qa.py        ← POST /api/qa/ask
│       ├── materials.py ← GET/DELETE /api/materials  GET /api/materials/{id}/summary
│       └── flashcards.py← POST /api/flashcards/generate  GET /api/flashcards/due
├── uploads/             ← Uploaded files saved here
├── data/
│   ├── learnspace.db    ← SQLite: materials, chunks, chat history, flashcards
│   └── chroma/          ← ChromaDB vector store (persisted to disk)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# ffmpeg (needed for audio/video transcription)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
```

### 2. Install dependencies

```bash
cd learnspace
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **First run note:** `sentence-transformers` will download the embedding model (~80 MB) and Whisper will download the `base` model (~150 MB) on first use. Both are cached locally after that.

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now running at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Offline Mode

LearnSpace works fully offline with [Ollama](https://ollama.ai):

```bash
# 1. Install Ollama from https://ollama.ai

# 2. Pull a model (3B is fast; 8B is smarter)
ollama pull llama3.2:3b

# 3. In your .env:
PREFER_OFFLINE=true
OLLAMA_MODEL=llama3.2:3b
```

When `ANTHROPIC_API_KEY` is empty or `PREFER_OFFLINE=true`, all LLM calls automatically route to Ollama.

---

## API Reference

### Ingest a file
```bash
curl -X POST http://localhost:8000/api/ingest/file \
  -F "file=@lecture.pdf"
# → { "material_id": "abc123", "status": "pending" }
```

### Check processing status
```bash
curl http://localhost:8000/api/ingest/status/abc123
# → { "status": "ready", "chunk_count": 87 }
```

### Ingest a YouTube video
```bash
curl -X POST http://localhost:8000/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=aircAruvnKk"}'
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is gradient descent?",
    "material_ids": ["abc123"]
  }'
# → { "answer": "...", "sources": [...], "session_id": "..." }
```

### Continue a conversation
```bash
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you give me an example?",
    "session_id": "the-session-id-from-above"
  }'
```

### Generate flashcards
```bash
curl -X POST http://localhost:8000/api/flashcards/generate/abc123 \
  -H "Content-Type: application/json" \
  -d '{"count": 15}'
```

### Get due flashcards (spaced repetition)
```bash
curl http://localhost:8000/api/flashcards/due
```

### Submit a review
```bash
curl -X POST http://localhost:8000/api/flashcards/{card_id}/review \
  -H "Content-Type: application/json" \
  -d '{"result": "correct"}'   # "correct" | "incorrect" | "skip"
```

### Get AI summary
```bash
curl http://localhost:8000/api/materials/abc123/summary
```

---

## Connecting to the Frontend

Point your frontend's API calls to `http://localhost:8000`. The CORS middleware allows all origins in development. For production, restrict `allow_origins` in `app/main.py`.

Example from the LearnSpace frontend HTML:

```javascript
const BASE = "http://localhost:8000/api";

// Ask a question
const res = await fetch(`${BASE}/qa/ask`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, material_ids: activeMaterialIds })
});
const { answer, sources, session_id } = await res.json();
```

---

## Supported Content Types

| Type | Formats | Method |
|------|---------|--------|
| PDF / Book | `.pdf`, `.epub` | PyMuPDF text extraction |
| Notes / Slides | `.docx`, `.pptx`, `.txt`, `.md` | python-docx / python-pptx |
| Video | `.mp4`, `.mov`, `.mkv`, `.webm` | Whisper transcription |
| Audio | `.mp3`, `.m4a`, `.wav` | Whisper transcription |
| YouTube | Any YouTube URL | Transcript API → Whisper fallback |
| Web article | Any URL | trafilatura clean extraction |

---

## Production Checklist

- [ ] Set `allow_origins` to your domain in `app/main.py`
- [ ] Use a proper secret key / auth middleware
- [ ] Run behind nginx or Caddy as a reverse proxy
- [ ] Use `gunicorn -k uvicorn.workers.UvicornWorker` for multi-worker production
- [ ] Back up `data/learnspace.db` and `data/chroma/` regularly
- [ ] Consider `WHISPER_MODEL=small` or `medium` for better transcription accuracy
