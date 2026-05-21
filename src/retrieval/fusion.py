"""
Reciprocal Rank Fusion (RRF) — combines dense and sparse retrieval results.

RRF formula (Cormack et al., 2009):
  RRF(d) = Σ_r [ 1 / (k + rank_r(d)) ]

where k=60 is a smoothing constant and rank_r(d) is the 1-indexed rank of
document d in ranker r. Higher RRF score = appeared near the top of more lists.

Why RRF over weighted score fusion?
  - Dense scores (cosine similarity) and sparse scores (BM25) live on completely
    different scales — averaging them directly is mathematically meaningless.
  - RRF uses only rank positions, making it robust to scale differences.
  - A document ranked #1 by both retrievers scores ~2/(k+1) ≈ 0.032.
  - k=60 prevents rank-1 documents from dominating — it smooths the contribution
    of top ranks vs. lower ranks, which is why 60 was chosen in the original paper.
  - Empirically outperforms weighted score fusion on BEIR and MTEB benchmarks.
"""
import logging

from src.config import get_settings
from src.retrieval.dense import retrieve_dense
from src.retrieval.sparse import retrieve_sparse

logger = logging.getLogger(__name__)
settings = get_settings()


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Fuse dense and sparse results using RRF.

    Preserves dense_score and sparse_score on each result so the full
    retrieval signal is visible in logs and the review UI.

    Returns:
      Fused list with rrf_score, dense_score, and sparse_score on each doc.
    """
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, dict] = {}
    dense_scores: dict[int, float] = {}
    sparse_scores: dict[int, float] = {}

    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc
        dense_scores[doc_id] = doc.get("dense_score", 0.0)

    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        sparse_scores[doc_id] = doc.get("sparse_score", 0.0)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, rrf_score in ranked[:top_k]:
        doc = {
            **doc_map[doc_id],
            "rrf_score":    round(rrf_score, 6),
            "dense_score":  round(dense_scores.get(doc_id, 0.0), 4),
            "sparse_score": round(sparse_scores.get(doc_id, 0.0), 4),
        }
        results.append(doc)

    return results


def retrieve_hybrid(query: str, top_k: int | None = None) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
      1. Dense retrieval  — BGE embeddings + pgvector cosine similarity
      2. Sparse retrieval — BM25 keyword matching
      3. RRF fusion       — rank-based combination of both lists

    Returns top_k fused results, each with rrf_score, dense_score, sparse_score.
    """
    final_k = top_k or settings.top_k_final

    dense_results  = retrieve_dense(query,  top_k=settings.top_k_dense)
    sparse_results = retrieve_sparse(query, top_k=settings.top_k_sparse)

    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=final_k,
        k=settings.rrf_k,
    )

    logger.debug(
        f"Hybrid retrieval: dense={len(dense_results)}, "
        f"sparse={len(sparse_results)}, fused={len(fused)}"
    )
    return fused
