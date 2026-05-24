"""
FastAPI application — exposes HybridRAG classifier as a REST API.

Endpoints:
  POST /classify        — classify a pharma marketing claim
  POST /ingest          — ingest documents into the knowledge base
  GET  /health          — database + model health check
  GET  /labels          — list valid compliance verdict labels
  GET  /review-queue    — low-confidence predictions needing human review

Security:
  - Rate limiting via SlowAPI (30 req/min on /classify, 10 req/min on /ingest)
  - Prompt injection + jailbreak detection on all text inputs
  - Input length bounds (10–2,000 chars per claim)
  - Ingest batch size cap (50 docs max)
  - Source domain allowlist on /ingest
  - Strict security response headers (HSTS, X-Frame-Options, CSP, etc.)
  - CORS locked to known origins in production (configurable via ALLOWED_ORIGINS env var)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from src.api.security import limiter, sanitize_claim, validate_ingest_docs
from src.classifier.classify import classify
from src.config import get_settings
from src.database import check_connection, get_db
from src.ingestion.ingest import ingest_documents
from src.retrieval.sparse import refresh_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

# ── CORS origins ──────────────────────────────────────────────────────────────
# Production: set ALLOWED_ORIGINS env var to a comma-separated list of exact
# origins (e.g. "https://yourportfolio.com,https://hybridrag.onrender.com").
# Development fallback: allow localhost ports and file:// for the portfolio HTML.
_env_origins = os.getenv("ALLOWED_ORIGINS", "")
if _env_origins.strip():
    CORS_ORIGINS: list[str] = [o.strip() for o in _env_origins.split(",") if o.strip()]
else:
    # Local dev — allow any localhost port and file:// protocol
    CORS_ORIGINS = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",   # file:// origins appear as "null" to the browser
    ]

logger.info(f"CORS origins: {CORS_ORIGINS}")


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BM25 index on startup from existing documents."""
    logger.info("Building BM25 index from existing documents...")
    refresh_bm25_index()
    logger.info("Startup complete.")
    yield


# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HybridRAG — Pharma MLR Claim Classifier",
    description="RAG-based compliance classification with dual retrieval and confidence scoring.",
    version="1.0.0",
    lifespan=lifespan,
    # Disable /docs and /redoc in production by checking an env flag
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    redoc_url=None,
)

# ── Rate limiter state ────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# ── Security headers middleware ───────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Attach hardened HTTP security headers to every response.

    Headers set:
      X-Content-Type-Options     — prevent MIME-sniffing
      X-Frame-Options            — block clickjacking (iframes)
      X-XSS-Protection           — legacy XSS filter (belt + suspenders)
      Strict-Transport-Security  — HTTPS-only for 1 year (with subdomains)
      Content-Security-Policy    — restrict resource origins
      Referrer-Policy            — don't leak URL in Referer header
      Permissions-Policy         — disable browser features we don't need
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]    = "nosniff"
    response.headers["X-Frame-Options"]           = "DENY"
    response.headers["X-XSS-Protection"]          = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"]   = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Referrer-Policy"]           = "no-referrer"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    return response


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    documents: list[dict[str, Any]]
    """Each doc: {"text": str, "label": str, "metadata": dict (optional)}"""

class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    failed: int
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
    """Check API liveness — database connectivity and label list."""
    db_ok = check_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database=db_ok,
        labels=settings.label_list,
    )


@app.get("/labels")
def get_labels() -> list[str]:
    """List all valid MLR compliance verdict labels."""
    return settings.label_list


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit("30/minute")
def classify_text(request: Request, body: ClassifyRequest):
    """
    Classify a pharmaceutical marketing claim.

    Rate limit: 30 requests/minute per IP.
    Input: plain claim text (10–2,000 characters).
    Output: compliance verdict + confidence + rationale + retrieved precedents.
    """
    # Sanitize — raises 400 on injection, length violation, or control chars
    clean_text = sanitize_claim(body.text)

    result = classify(clean_text, persist=True)

    # Strip embedding vectors — too large to serialise and not needed by clients
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


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
def ingest(request: Request, body: IngestRequest):
    """
    Ingest validated documents into the knowledge base.

    Rate limit: 10 requests/minute per IP.
    Input: batch of up to 50 documents with text + label + optional metadata.
    Sources must originate from trusted FDA / NLM domains.
    """
    # Validate + sanitise — raises 400 on bad labels, domains, injection, etc.
    clean_docs = validate_ingest_docs(body.documents)

    counts = ingest_documents(clean_docs)
    refresh_bm25_index()

    return IngestResponse(
        ingested=counts["ingested"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        message=(
            f"Ingested {counts['ingested']} document(s). "
            f"Skipped {counts['skipped']} duplicate(s). "
            f"BM25 index refreshed."
        ),
    )


@app.get("/review-queue")
def review_queue(limit: int = 20) -> list[dict]:
    """Return recent low-confidence predictions that need human review."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")
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
