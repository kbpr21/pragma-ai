-- pragma SQLite schema - Migration 003: Embedding storage
-- Stores fact-level vector embeddings for semantic retrieval.

CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_embeddings_model ON fact_embeddings(model);
