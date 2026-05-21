"""
Database connection and session management.
Uses SQLAlchemy + pgvector.

This is the only file in the project that talks to Postgres directly.
All other modules call the functions defined here.
"""
import json
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,     # checks connections are alive before use
    pool_size=5,            # persistent connections in the pool
    max_overflow=10,        # extra connections allowed under burst load
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Commits on success, rolls back on any exception, always closes.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_connection() -> bool:
    """Returns True if the database is reachable."""
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def insert_document(db: Session, content: str, label: str, metadata: dict = None) -> int:
    """Insert a document and return its new ID."""
    result = db.execute(
        text("""
            INSERT INTO documents (text, label, metadata)
            VALUES (:text, :label, :metadata)
            RETURNING id
        """),
        {"text": content, "label": label, "metadata": json.dumps(metadata or {})}
    )
    return result.scalar_one()


def insert_embedding(db: Session, doc_id: int, embedding: list[float]) -> None:
    """
    Store a dense embedding for a document.
    ON CONFLICT DO NOTHING respects the UNIQUE(doc_id) constraint —
    re-ingesting a document skips the duplicate embedding silently.
    """
    db.execute(
        text("""
            INSERT INTO embeddings (doc_id, embedding)
            VALUES (:doc_id, :embedding)
            ON CONFLICT (doc_id) DO NOTHING
        """),
        {"doc_id": doc_id, "embedding": str(embedding)}
    )


def search_dense(db: Session, query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Cosine similarity search via pgvector.
    Returns top_k documents ordered by similarity (highest first).

    The <=> operator returns cosine distance (0 = identical, 2 = opposite).
    Score = 1 - distance, so higher score = more similar.
    The double-cast (:query_embedding)::vector is required by SQLAlchemy —
    parentheses prevent the colon from being parsed as a named parameter prefix.
    """
    results = db.execute(
        text("""
            SELECT
                d.id,
                d.text,
                d.label,
                d.metadata,
                1 - (e.embedding <=> (:query_embedding)::vector) AS score
            FROM embeddings e
            JOIN documents d ON d.id = e.doc_id
            ORDER BY e.embedding <=> (:query_embedding)::vector
            LIMIT :top_k
        """),
        {"query_embedding": str(query_embedding), "top_k": top_k}
    )
    return [dict(row._mapping) for row in results]


def get_all_documents(db: Session) -> list[dict]:
    """Fetch all documents — used by BM25 to build its in-memory index at startup."""
    results = db.execute(
        text("SELECT id, text, label, metadata FROM documents ORDER BY id")
    )
    return [dict(row._mapping) for row in results]


def log_classification(
    db: Session,
    input_text: str,
    latency_ms: int,
    status: str = "completed",
    predicted_label: str | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
    retrieved_docs: list[dict] | None = None,
    error_message: str | None = None,
) -> int:
    """
    Persist a classification attempt to the log table.

    Works for both successful classifications and failures:
      - On success: pass predicted_label, confidence, rationale, retrieved_docs
      - On failure: pass status='failed', error_message, leave others None
    """
    result = db.execute(
        text("""
            INSERT INTO classification_log
                (input_text, predicted_label, confidence, rationale,
                 retrieved_docs, latency_ms, status, error_message)
            VALUES
                (:input_text, :predicted_label, :confidence, :rationale,
                 :retrieved_docs, :latency_ms, :status, :error_message)
            RETURNING id
        """),
        {
            "input_text":      input_text,
            "predicted_label": predicted_label,
            "confidence":      confidence,
            "rationale":       rationale,
            "retrieved_docs":  json.dumps(retrieved_docs) if retrieved_docs else None,
            "latency_ms":      latency_ms,
            "status":          status,
            "error_message":   error_message,
        }
    )
    return result.scalar_one()
