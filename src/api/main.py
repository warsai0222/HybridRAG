"""
FastAPI application — exposes HybridRAG classifier as a REST API.

Endpoints:
  POST /ingest          — ingest documents into the knowledge base
  POST /classify        — classify a pharma marketing claim
  GET  /health          — database + model health check
  GET  /labels          — list valid compliance verdict labels
  GET  /review-queue    — low-confidence predictions needing human review
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

from src.classifier.classify import classify
from src.config import get_settings
from src.database import check_connection, get_db
from src.ingestion.ingest import ingest_documents
from src.retrieval.sparse import refresh_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BM25 index on startup from existing documents."""
    logger.info("Building BM25 index from existing documents...")
    refresh_bm25_index()
    logger.info("Startup complete.")
    yield


app = FastAPI(
    title="HybridRAG — Pharma MLR Claim Classifier",
    description="RAG-based compliance classification with dual retrieval and confidence scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the portfolio HTML (opened as file://) and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    documents: list[dict[str, Any]]
    """Each doc: {"text": str, "label": str, "metadata": dict (optional)}"""

class IngestResponse(BaseModel):
    ingested: int
    message: str

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    rationale: str
    needs_human_review: bool
    latency_ms: int
    retrieved_examples: list[dict]

class HealthResponse(BaseModel):
    status: str
    database: bool
    labels: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    db_ok = check_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database=db_ok,
        labels=settings.label_list,
    )


@app.get("/labels")
def get_labels() -> list[str]:
    return settings.label_list


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    n = ingest_documents(request.documents)
    refresh_bm25_index()

    return IngestResponse(
        ingested=n,
        message=f"Successfully ingested {n} documents. BM25 index refreshed.",
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify_text(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = classify(request.text, persist=True)

    # Strip embedding vectors from response — too large to serialize
    examples = [
        {k: v for k, v in ex.items() if k != "embedding"}
        for ex in result.retrieved_examples
    ]

    return ClassifyResponse(
        label=result.label,
        confidence=result.confidence,
        rationale=result.rationale,
        needs_human_review=result.needs_human_review,
        latency_ms=result.latency_ms,
        retrieved_examples=examples,
    )


@app.get("/review-queue")
def review_queue(limit: int = 20) -> list[dict]:
    """Return recent low-confidence predictions that need human review."""
    with get_db() as db:
        rows = db.execute(
            text("""
                SELECT id, input_text, predicted_label, confidence,
                       rationale, latency_ms, created_at
                FROM classification_log
                WHERE confidence < :threshold
                  AND status = 'completed'
                  AND reviewed_at IS NULL
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"threshold": 0.7, "limit": limit}
        )
        return [dict(row._mapping) for row in rows]
