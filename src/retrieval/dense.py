"""
Dense retrieval using BGE embeddings + pgvector cosine similarity search.

Semantic retrieval — understands meaning even when exact words differ.
e.g. "reduces cardiac events" and "lowers heart attack risk" score high together.
"""
import logging

from src.database import get_db, search_dense
from src.ingestion.ingest import embed_query

logger = logging.getLogger(__name__)


def retrieve_dense(query: str, top_k: int = 10) -> list[dict]:
    """
    Retrieve the top_k most semantically similar documents to the query.

    Embeds the query with the BGE retrieval prefix, then runs cosine similarity
    search against all stored document vectors via pgvector's HNSW index.

    Returns:
      [{"id": int, "text": str, "label": str, "metadata": dict, "dense_score": float}]
      dense_score is cosine similarity — higher = more semantically similar.
    """
    query_embedding = embed_query(query)

    with get_db() as db:
        results = search_dense(db, query_embedding=query_embedding, top_k=top_k)

    # Rename 'score' → 'dense_score' so it's distinguishable after fusion
    for r in results:
        r["dense_score"] = r.pop("score", None)

    logger.debug(f"Dense retrieval: {len(results)} results for query='{query[:60]}'")
    return results
