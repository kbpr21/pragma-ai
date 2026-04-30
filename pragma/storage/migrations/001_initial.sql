-- pragma SQLite schema - Initial migration

-- Core tables

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    char_count  INTEGER,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT,
    aliases     TEXT,
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id           TEXT PRIMARY KEY,
    subject_id   TEXT,
    predicate    TEXT NOT NULL,
    object_id    TEXT,
    object_value TEXT,
    context      TEXT,
    source_doc   TEXT,
    source_page  INTEGER,
    confidence   REAL DEFAULT 1.0,
    ingested_at  TEXT NOT NULL,
    valid_from   TEXT,
    valid_until  TEXT,
    superseded_by TEXT,
    is_active    INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS relationships (
    id           TEXT PRIMARY KEY,
    from_entity  TEXT,
    rel_type    TEXT NOT NULL,
    to_entity   TEXT,
    fact_ids    TEXT,
    weight     REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS query_cache (
    id           TEXT PRIMARY KEY,
    query_hash   TEXT NOT NULL UNIQUE,
    query_text   TEXT NOT NULL,
    answer      TEXT NOT NULL,
    reasoning   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    ttl_seconds INTEGER DEFAULT 3600
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(is_active);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_entity);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_entity);
