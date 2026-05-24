-- HybridRAG — PostgreSQL schema initialisation
-- Runs automatically when the pgvector container boots for the first time.
-- Safe to re-run: all statements use IF NOT EXISTS.

-- ── pgvector extension ────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── documents ─────────────────────────────────────────────────────────────────
-- Stores the raw text of each ingested pharma claim example plus its MLR label.
-- metadata is free-form JSONB for source info (e.g. FDA letter ID, drug name).
-- content_hash (SHA-256) deduplicates on re-ingestion — same text is never
-- inserted twice even if the source URL changes.
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    text         TEXT        NOT NULL,
    label        TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}',
    content_hash TEXT        UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── ingested_urls ──────────────────────────────────────────────────────────────
-- Tracks every source URL that has been scraped and processed.
-- The scraper checks this table before downloading a PDF — already-seen URLs
-- are skipped entirely, making incremental runs safe and cheap.
-- status: 'ingested' | 'skipped_dedup' | 'failed' | 'review_queued'
CREATE TABLE IF NOT EXISTS ingested_urls (
    id          SERIAL PRIMARY KEY,
    source_url  TEXT        NOT NULL UNIQUE,
    status      TEXT        NOT NULL DEFAULT 'ingested',
    doc_count   INTEGER     NOT NULL DEFAULT 0,
    notes       TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── review_queue ───────────────────────────────────────────────────────────────
-- Documents that passed rule-based validation but need a human to confirm the
-- label before they enter the live knowledge base.
-- This is mandatory for 'supported' labels from non-enforcement sources (e.g. PI
-- documents) because a false positive 'supported' verdict carries regulatory risk.
-- status: 'pending' | 'approved' | 'rejected'
CREATE TABLE IF NOT EXISTS review_queue (
    id           SERIAL PRIMARY KEY,
    text         TEXT        NOT NULL,
    label        TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}',
    content_hash TEXT        NOT NULL UNIQUE,
    reason       TEXT,               -- why it was queued (e.g. "supported from PI source")
    status       TEXT        NOT NULL DEFAULT 'pending',
    reviewed_by  TEXT,
    reviewed_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── embeddings ────────────────────────────────────────────────────────────────
-- Dense vector for each document (BGE-large-en, dim=1024).
-- UNIQUE(doc_id) prevents duplicate embeddings on re-ingest.
-- The ivfflat index enables fast approximate nearest-neighbour search via <=>
-- (cosine distance). lists=100 is a sensible default for up to ~1M vectors.
CREATE TABLE IF NOT EXISTS embeddings (
    id        SERIAL PRIMARY KEY,
    doc_id    INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    UNIQUE (doc_id)
);

CREATE INDEX IF NOT EXISTS embeddings_ivfflat_idx
    ON embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ── classification_log ────────────────────────────────────────────────────────
-- Audit trail for every classification attempt (success or failure).
-- retrieved_docs stores the top-k examples returned by the retriever as JSONB.
CREATE TABLE IF NOT EXISTS classification_log (
    id               SERIAL PRIMARY KEY,
    input_text       TEXT,
    predicted_label  TEXT,
    confidence       FLOAT,
    rationale        TEXT,
    retrieved_docs   JSONB,
    latency_ms       INTEGER,
    status           TEXT        NOT NULL DEFAULT 'completed',
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
