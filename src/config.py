"""
Central configuration — loaded once at startup from .env

All secrets (API keys, DB credentials) live in .env only.
config.py defines the shape and defaults; .env provides the values.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://hybridrag:hybridrag@localhost:5432/hybridrag"

    # ── Groq (OpenAI-compatible) ──────────────────────────────────────────────
    # Groq's API is a drop-in replacement for OpenAI — same client, same interface.
    # Only the base_url and api_key change. The model runs on Groq's hardware for free.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Dense and sparse each cast a wider net (10) so RRF has real reranking to do.
    # Final output is trimmed to 5 after fusion.
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5
    rrf_k: int = 60             # smoothing constant from the original RRF paper

    # Minimum RRF score for the top retrieved result to be considered usable evidence.
    # With k=60, a document ranked #1 in both lists scores ~0.033 and ranked #10
    # in both scores ~0.029. A score below 0.015 means nothing in the KB is
    # sufficiently similar — return insufficient_data rather than hallucinate.
    retrieval_quality_threshold: float = 0.015

    # ── Pharma MLR compliance labels ──────────────────────────────────────────
    labels: str = (
        "supported,"
        "unsupported,"
        "partially_supported,"
        "false_balance,"
        "needs_legal_review,"
        "insufficient_data"
    )

    @property
    def label_list(self) -> list[str]:
        return [label.strip() for label in self.labels.split(",")]


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    lru_cache means .env is read once at startup — not on every function call.
    """
    return Settings()
