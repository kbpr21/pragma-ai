"""Latent semantic retrieval using sentence embeddings.

Provides fuzzy recall for queries that BM25 keyword matching misses —
synonyms, paraphrases, conceptual similarity. Uses ``sentence-transformers``
when available; degrades gracefully to a no-op when the dependency is not
installed (the ``[embeddings]`` optional extra).

Design constraints:
- Zero infrastructure: pure NumPy cosine similarity, no external vector DB
- Incremental: new facts are appended without rebuilding the full index
- Serialisation: embeddings are persisted to SQLite as BLOB (numpy bytes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Vector-similarity retrieval over fact contexts.

    Lazy-loads ``sentence-transformers`` so the import cost is only paid
    when the caller actually invokes retrieval.  If the library is
    missing the retriever returns empty results and logs a warning.
    """

    def __init__(
        self,
        storage: Any,
        model_name: str = "all-MiniLM-L6-v2",
        kb_dir: Optional[str] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.storage = storage
        self.model_name = model_name
        self._kb_dir = Path(kb_dir) if kb_dir else None
        self._llm = llm
        self._model: Any = None
        self._available: Optional[bool] = None
        # In-memory cache: {fact_id: embedding_vector}
        self._index: Dict[str, Any] = {}
        self._index_loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        """Load the embedding model or verify the LLM provider's embedding method.

        Returns True if a model or LLM provider's embedding capability is usable.
        """
        if self._available is not None:
            return self._available

        # Check if the LLM provider has an embed method
        if self._llm is not None and hasattr(self._llm, "embed"):
            self._available = True
            logger.info("SemanticRetriever: using API-based embedding via LLM provider")
            return True

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._available = True
            logger.info("SemanticRetriever: loaded model %s", self.model_name)
        except ImportError:
            self._available = False
            logger.warning(
                "sentence-transformers not installed and LLM provider lacks embed capability; "
                "semantic retrieval disabled. Install with: "
                "pip install 'pragma-ai[embeddings]'"
            )
        except Exception as e:  # noqa: BLE001
            self._available = False
            logger.warning("Failed to load embedding model: %s", e)
        return self._available

    @property
    def available(self) -> bool:
        """True when the embedding model is usable."""
        return self._ensure_model()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load persisted embeddings from SQLite into memory."""
        if self._index_loaded:
            return
        self._index_loaded = True
        try:
            import numpy as np

            conn = self.storage._get_connection()
            rows = conn.execute(
                "SELECT fact_id, embedding FROM fact_embeddings WHERE model = ?",
                (self.model_name,),
            ).fetchall()
            for row in rows:
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                self._index[row["fact_id"]] = vec
            if rows:
                logger.debug(
                    "SemanticRetriever: loaded %d embeddings from DB", len(rows)
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not load embeddings index: %s", e)

    def embed_facts(self, facts: List[Dict[str, Any]]) -> int:
        """Compute and persist embeddings for a list of fact dicts.

        Only embeds facts that are not already in the index.

        Args:
            facts: List of fact dicts with at least ``id`` and ``context``.

        Returns:
            Number of newly embedded facts.
        """
        if not self._ensure_model():
            return 0
        if not facts:
            return 0

        import numpy as np
        from datetime import datetime, timezone

        self._load_index()

        # Filter to facts not yet embedded.
        new_facts = [f for f in facts if f.get("id") and f["id"] not in self._index]
        if not new_facts:
            return 0

        # Build texts to embed.  Prefer context (full sentence); fall
        # back to a compact triple representation.
        texts = []
        for f in new_facts:
            text = f.get("context", "").strip()
            if not text:
                subj = f.get("subject", f.get("subject_id", ""))
                pred = f.get("predicate", "")
                obj = f.get("object_value") or f.get("object", f.get("object_id", ""))
                text = f"{subj} {pred} {obj}".strip()
            texts.append(text or "unknown")

        try:
            if self._llm is not None and hasattr(self._llm, "embed"):
                raw_embeddings = self._llm.embed(texts, model=self.model_name)
                # Convert to numpy arrays and L2-normalize
                embeddings = []
                for e in raw_embeddings:
                    vec = np.array(e, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    embeddings.append(vec)
            else:
                raw_embeddings = self._model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=64,
                    normalize_embeddings=True,
                )
                embeddings = [e for e in raw_embeddings]
        except Exception as e:  # noqa: BLE001
            logger.warning("Embedding computation failed: %s", e)
            return 0

        # Persist to SQLite and update in-memory index.
        conn = self.storage._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        for fact, vec in zip(new_facts, embeddings):
            fact_id = fact["id"]
            vec_f32 = vec.astype(np.float32)
            self._index[fact_id] = vec_f32
            conn.execute(
                "INSERT OR REPLACE INTO fact_embeddings "
                "(fact_id, embedding, model, created_at) VALUES (?, ?, ?, ?)",
                (fact_id, vec_f32.tobytes(), self.model_name, now),
            )
        conn.commit()
        logger.debug("SemanticRetriever: embedded %d new facts", len(new_facts))
        return len(new_facts)

    def query(
        self,
        text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find the most semantically similar facts to a query string.

        Args:
            text: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of ``(fact_id, cosine_similarity)`` tuples, sorted
            descending by similarity.
        """
        if not self._ensure_model():
            return []

        import numpy as np

        self._load_index()
        if not self._index:
            return []

        try:
            if self._llm is not None and hasattr(self._llm, "embed"):
                raw_query_vec = self._llm.embed([text], model=self.model_name)[0]
                query_vec = np.array(raw_query_vec, dtype=np.float32)
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
            else:
                query_vec = self._model.encode(
                    [text],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )[0].astype(np.float32)
        except Exception as e:  # noqa: BLE001
            logger.warning("Query embedding failed: %s", e)
            return []

        # Cosine similarity (vectors are already L2-normalised).
        fact_ids = list(self._index.keys())
        matrix = np.stack([self._index[fid] for fid in fact_ids])
        sims = matrix @ query_vec

        # Top-k by similarity.
        if len(sims) <= top_k:
            top_indices = list(range(len(sims)))
        else:
            top_indices = list(np.argpartition(sims, -top_k)[-top_k:])
        top_indices.sort(key=lambda i: sims[i], reverse=True)

        return [(fact_ids[i], float(sims[i])) for i in top_indices]
