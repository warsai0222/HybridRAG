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


def ingest_documents(documents: list[dict[str, Any]], batch_size: int = 32) -> int:
    """
    Ingest a list of documents into PostgreSQL.

    Each document must have:
      - text  (str): the document content (e.g., a pharma marketing claim)
      - label (str): ground-truth compliance verdict
      - metadata (dict, optional): any extra fields (source, reviewer, date, etc.)

    Batches texts before embedding — the model processes a batch in parallel,
    which is dramatically faster than encoding one document at a time.

    Returns the number of documents successfully ingested.
    """
    if not documents:
        logger.warning("ingest_documents called with empty list")
        return 0

    ingested = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc["text"] for doc in batch]

        logger.info(f"Embedding batch {i // batch_size + 1} ({len(texts)} docs)...")
        embeddings = embed(texts)

        with get_db() as db:
            for doc, embedding in zip(batch, embeddings):
                doc_id = insert_document(
                    db,
                    content=doc["text"],
                    label=doc["label"],
                    metadata=doc.get("metadata", {}),
                )
                insert_embedding(db, doc_id=doc_id, embedding=embedding)
                ingested += 1

        logger.info(f"Ingested {ingested}/{len(documents)} documents")

    return ingested


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
