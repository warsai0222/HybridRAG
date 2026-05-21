"""
Sparse retrieval using BM25 (Okapi BM25).

Keyword retrieval — excels at exact term matches that dense embeddings miss.
e.g. specific drug names, regulatory codes, ICD codes, citation numbers.

The index is rebuilt in-memory at startup from the full document corpus.
For corpora > 100k documents, replace with Elasticsearch or OpenSearch.
"""
import logging
import re

from rank_bm25 import BM25Okapi

from src.database import get_db, get_all_documents

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract word tokens. Swap for a domain tokenizer if needed."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Index:
    """
    In-memory BM25 index built from all documents in the database.
    Call .refresh() to rebuild after new documents are ingested.
    """

    def __init__(self):
        self._docs: list[dict] = []
        self._index: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the index from the current document corpus."""
        with get_db() as db:
            self._docs = get_all_documents(db)

        if not self._docs:
            logger.warning("BM25 index: no documents found — index is empty")
            self._index = None
            return

        tokenized_corpus = [_tokenize(doc["text"]) for doc in self._docs]
        self._index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built: {len(self._docs)} documents")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top_k documents by BM25 score.

        BM25 scores are not normalized — they live on a different scale than
        cosine similarity scores from dense retrieval. This is why RRF uses
        rank positions rather than raw scores for fusion.

        Returns:
          [{"id": int, "text": str, "label": str, "metadata": dict, "sparse_score": float}]
        """
        if self._index is None or not self._docs:
            logger.warning("BM25 search called on empty index — returning empty results")
            return []

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        scored = sorted(
            zip(self._docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in scored[:top_k]:
            results.append({**doc, "sparse_score": float(score)})

        logger.debug(f"BM25 retrieval: {len(results)} results for query='{query[:60]}'")
        return results


# Module-level singleton — shared across all requests
_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def retrieve_sparse(query: str, top_k: int = 10) -> list[dict]:
    """Public interface: BM25 retrieval."""
    return get_bm25_index().search(query, top_k=top_k)


def refresh_bm25_index() -> None:
    """Call after ingesting new documents to keep the in-memory index current."""
    get_bm25_index().refresh()
