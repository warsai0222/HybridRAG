"""
Document ingestion pipeline.

Flow:
  raw text + label → embed (BGE) → store in PostgreSQL (documents + embeddings tables)

Usage:
  from src.ingestion.ingest import ingest_documents
  ingest_documents([{"text": "...", "label": "supported", "metadata": {...}}])
"""
import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.database import get_db, insert_document, insert_embedding

logger = logging.getLogger(__name__)
settings = get_settings()

# Singleton — BGE is ~1.3 GB. Load once at startup, reuse for every call.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Compute BGE embeddings for a list of documents.

    No prefix is added — BGE was trained to embed documents as plain text.
    Only queries get the retrieval prefix (see embed_query below).

    normalize_embeddings=True scales each vector to unit length so that
    cosine similarity == dot product, and all scores are consistently in [-1, 1].
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
    )
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """
    Embed a search query with the BGE retrieval prefix.

    BGE was trained with an asymmetric setup:
      - Documents → embedded as-is
      - Queries   → prefixed with "Represent this sentence for searching relevant passages:"

    The prefix signals to the model that this text is a query seeking relevant documents,
    not a document to be indexed. Using the wrong function silently degrades retrieval quality.
    Always call embed_query() at retrieval time, embed() at ingestion time.
    """
    model = _get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()


def ingest_documents(documents: list[dict[str, Any]], batch_size: int = 32) -> dict:
    """
    Ingest a list of documents into PostgreSQL with deduplication.

    Each document must have:
      - text  (str): the document content (e.g., a pharma marketing claim)
      - label (str): ground-truth compliance verdict
      - metadata (dict, optional): any extra fields (source, reviewer, date, etc.)

    Deduplication:
      - content_hash (SHA-256) is computed for each doc before insert
      - Documents already in the DB (same hash) are silently skipped
      - This makes the function safe to call repeatedly on the same JSONL

    Batches texts before embedding — the model processes a batch in parallel,
    which is dramatically faster than encoding one document at a time.

    Returns a dict with ingested / skipped / failed counts.
    """
    import hashlib, re
    from src.database import document_exists_by_hash

    def _hash(text: str) -> str:
        norm = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(norm.encode()).hexdigest()

    if not documents:
        logger.warning("ingest_documents called with empty list")
        return {"ingested": 0, "skipped": 0, "failed": 0}

    counts = {"ingested": 0, "skipped": 0, "failed": 0}

    # Pre-filter duplicates in a single DB pass before embedding (saves GPU time)
    to_embed: list[dict] = []
    hashes:   list[str]  = []
    with get_db() as db:
        for doc in documents:
            h = doc.get("_content_hash") or _hash(doc["text"])
            if document_exists_by_hash(db, h):
                counts["skipped"] += 1
            else:
                to_embed.append(doc)
                hashes.append(h)

    if counts["skipped"]:
        logger.info(f"Dedup: skipped {counts['skipped']} already-ingested documents")

    if not to_embed:
        logger.info("Nothing new to ingest after dedup check")
        return counts

    for i in range(0, len(to_embed), batch_size):
        batch       = to_embed[i : i + batch_size]
        batch_hash  = hashes[i : i + batch_size]
        texts       = [doc["text"] for doc in batch]

        logger.info(f"Embedding batch {i // batch_size + 1} ({len(texts)} docs)...")
        embeddings = embed(texts)

        with get_db() as db:
            for doc, embedding, h in zip(batch, embeddings, batch_hash):
                try:
                    doc_id = insert_document(
                        db,
                        content=doc["text"],
                        label=doc["label"],
                        metadata=doc.get("metadata", {}),
                        content_hash=h,
                    )
                    insert_embedding(db, doc_id=doc_id, embedding=embedding)
                    counts["ingested"] += 1
                except Exception as e:
                    logger.error(f"Failed to insert doc (hash={h[:12]}): {e}")
                    counts["failed"] += 1

        logger.info(
            f"Progress: ingested={counts['ingested']}, "
            f"skipped={counts['skipped']}, failed={counts['failed']}"
        )

    return counts


def ingest_from_jsonl(path: str) -> int:
    """
    Ingest documents from a JSONL file.
    Each line: {"text": "...", "label": "...", "metadata": {...}}
    """
    import json
    documents = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    logger.info(f"Loaded {len(documents)} documents from {path}")
    return ingest_documents(documents)
