-- pragma SQLite schema - Migration 002: Semantic metadata
-- Adds epistemic modality tracking to facts for nuance preservation.

-- New columns on facts table (idempotent via try/catch in Python).
-- SQLite does not support IF NOT EXISTS on ALTER TABLE, so the
-- Python migration runner wraps each statement in a try/except.

ALTER TABLE facts ADD COLUMN modality TEXT DEFAULT 'assertion';
ALTER TABLE facts ADD COLUMN is_speculative INTEGER DEFAULT 0;
ALTER TABLE facts ADD COLUMN hedge_phrase TEXT;

-- Performance index for modality-filtered queries.
CREATE INDEX IF NOT EXISTS idx_facts_modality ON facts(modality);
CREATE INDEX IF NOT EXISTS idx_facts_speculative ON facts(is_speculative);
