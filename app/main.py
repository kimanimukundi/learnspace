"""
LearnSpace Backend — Main FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from .routers import ingest, qa, materials, flashcards
from .db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("LearnSpace backend starting up…")
    await init_db()
    yield
    logger.info("LearnSpace backend shutting down.")


app = FastAPI(
    title="LearnSpace API",
    description="Personal AI learning assistant — ingest, index, and query your study materials.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (dev: allow all; prod: lock to your domain) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router,     prefix="/api/ingest",     tags=["Ingest"])
app.include_router(qa.router,         prefix="/api/qa",         tags=["Q&A"])
app.include_router(materials.router,  prefix="/api/materials",  tags=["Materials"])
app.include_router(flashcards.router, prefix="/api/flashcards", tags=["Flashcards"])

# ── Serve the frontend HTML (optional convenience) ────────────────────────────
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")


@app.get("/api/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}
