import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from pragma.config import PragmaConfig
from pragma.ingestion.extractor import FactExtractor
from pragma.ingestion.loader import DocumentLoader
from pragma.ingestion.preprocessor import DocumentPreprocessor
from pragma.llm import get_provider
from pragma.llm.base import LLMProvider
from pragma.models import AtomicFact, KBStats, PragmaResult, ReasoningStep
from pragma.graph.builder import GraphBuilder
from pragma.graph.resolver import EntityResolver
from pragma.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Main interface for the pragma knowledge base."""

    def __init__(
        self,
        llm: Optional[Union[str, LLMProvider]] = None,
        kb_dir: str = "./pragma_kb",
        config: Optional[PragmaConfig] = None,
    ) -> None:
        self.config = config or PragmaConfig(kb_dir=kb_dir)
        self._kb_dir = Path(self.config.kb_dir)
        self._kb_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(llm, str):
            self._llm = get_provider(llm)
        elif llm is None:
            self._llm = get_provider(self.config.llm_provider)
        else:
            self._llm = llm

        self._storage = SQLiteStore(kb_dir=kb_dir)
        self._loader = DocumentLoader()
        self._preprocessor = DocumentPreprocessor(
            max_tokens=self.config.max_facts_per_segment
        )
        self._extractor = FactExtractor(self._llm)
        self._resolver = EntityResolver(self._storage, fuzzy_threshold=85)
        self._graph_builder = GraphBuilder(self._storage, kb_dir=kb_dir)

    @classmethod
    def from_config(cls, path: str) -> "KnowledgeBase":
        """Create KnowledgeBase from config file."""
        config = PragmaConfig.from_yaml(path)
        return cls(config=config)

    @property
    def llm(self) -> LLMProvider:
        """Get the LLM provider."""
        return self._llm

    @property
    def kb_dir(self) -> Path:
        """Get the knowledge base directory."""
        return self._kb_dir

    def stats(self) -> KBStats:
        """Get knowledge base statistics."""
        return self._storage.get_kb_stats()

    def ingest(
        self,
        source: Union[str, Path, List[str], dict],
        show_progress: bool = True,
    ) -> "IngestResult":
        """Ingest documents into the knowledge base.

        Pipeline:
        1. Load document(s) → DocumentLoader
        2. Preprocess segments → DocumentPreprocessor
        3. Extract facts (LLM) → FactExtractor
        4. Resolve entities → EntityResolver
        5. Build graph → GraphBuilder

        Args:
            source: File path, directory, URL, list, or dict
            show_progress: Show progress bar (requires rich)

        Returns:
            IngestResult with summary statistics
        """
        if isinstance(source, (list, dict)):
            sources = source if isinstance(source, list) else [source]
        else:
            source = Path(source) if not isinstance(source, str) else source
            if Path(source).is_dir():
                sources = self._discover_files(source)
            else:
                sources = [source]

        return self._ingest_batch(sources, show_progress=show_progress)

    def _discover_files(self, directory: Path) -> List[Path]:
        """Discover all supported files in a directory."""
        supported_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".csv",
            ".json",
            ".jsonl",
            ".docx",
            ".html",
            ".htm",
        }
        files = []
        for ext in supported_extensions:
            files.extend(directory.rglob(f"*{ext}"))
        return sorted(files)

    def _ingest_batch(
        self,
        sources: List[Union[str, Path, dict]],
        show_progress: bool = True,
    ) -> "IngestResult":
        """Ingest a batch of sources."""
        total_docs = 0
        total_facts = 0
        total_entities = 0
        skipped = 0

        for source in sources:
            try:
                result = self._ingest_single(source)
                total_docs += result.documents
                total_facts += result.facts
                total_entities += result.entities
                skipped += result.skipped
            except Exception as e:
                logger.warning(f"Failed to ingest {source}: {e}")

        self._graph_builder.save()

        return IngestResult(
            documents=total_docs,
            facts=total_facts,
            entities=total_entities,
            skipped=skipped,
        )

    def _ingest_single(self, source: Union[str, Path, dict]) -> "IngestResult":
        """Ingest a single source."""
        source_str = str(source)

        doc_id = self._compute_doc_id(source_str)

        if self._storage.document_exists(doc_id):
            logger.debug(f"Skipping already-ingested document: {source}")
            return IngestResult(documents=0, facts=0, entities=0, skipped=1)

        segments = self._loader.load(source)
        if not segments:
            return IngestResult(documents=0, facts=0, entities=0, skipped=0)

        self._storage.save_document(
            doc_id,
            source_str,
            segments[0].doc_type if segments else "unknown",
            sum(len(s.content) for s in segments),
        )

        processed = self._preprocessor.preprocess(segments)

        all_facts = []
        for i in range(0, len(processed), 10):
            batch = processed[i : i + 10]
            if len(batch) > 1:
                facts = self._extractor.extract_batch(batch, max_tokens=4000)
            else:
                facts = self._extractor.extract(batch)
            all_facts.extend(facts)

        fact_dicts = all_facts

        new_entities = set()
        for fact_dict in fact_dicts:
            subject = self._resolver.resolve(fact_dict["subject"], entity_type=None)
            new_entities.add(subject.id)

            object_id: Optional[str] = None
            if fact_dict.get("object"):
                obj = self._resolver.resolve(fact_dict["object"], entity_type=None)
                new_entities.add(obj.id)
                object_id = obj.id

            fact = AtomicFact(
                id=str(uuid.uuid4()),
                subject_id=subject.id,
                predicate=fact_dict["predicate"],
                object_id=object_id,
                object_value=fact_dict.get("object_value"),
                context=fact_dict.get("_context", ""),
                source_doc=fact_dict.get("_source_doc", ""),
                source_page=fact_dict.get("_source_page"),
                confidence=fact_dict.get("confidence", 1.0),
            )
            self._storage.save_fact(fact)
            self._graph_builder.add_entity(subject)
            self._graph_builder.add_fact(fact)

        return IngestResult(
            documents=1,
            facts=len(fact_dicts),
            entities=len(new_entities),
            skipped=0,
        )

    def _compute_doc_id(self, content: str) -> str:
        """Compute document ID.

        For real files, hash the file *contents* so the same file at two paths
        (or a moved file) deduplicates correctly. For URLs/strings/dicts, fall
        back to hashing the source string.
        """
        try:
            path = Path(content)
            if path.is_file():
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()[:16]
        except (OSError, ValueError):
            pass

        normalized = content.lower().strip()
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def query(
        self,
        query: str,
        hop_depth: Optional[int] = None,
        min_confidence: float = 0.5,
        as_of: Optional[str] = None,
        top_k: int = 5,
    ) -> "PragmaResult":
        """Query the knowledge base.

        Pipeline:
        1. Check query cache
        2. Decompose query into sub-questions
        3. Retrieve seed entities via BM25
        4. Traverse graph to extract subgraph
        5. Assemble facts from subgraph
        6. Synthesize answer via LLM

        Args:
            query: Natural language query
            hop_depth: Maximum graph traversal depth (default: config.default_hop_depth)
            min_confidence: Minimum confidence threshold for facts
            as_of: Temporal filter (not implemented yet)
            top_k: Number of results to return

        Returns:
            PragmaResult with answer, reasoning path, and metadata
        """
        from datetime import datetime

        start_time = time.time()

        cache_key = self._compute_query_cache_key(
            query, hop_depth, min_confidence, as_of
        )
        cached = self._storage.get_query_cache(cache_key)
        if cached:
            cached.latency_ms = (time.time() - start_time) * 1000
            return cached

        if hop_depth is None:
            hop_depth = self.config.default_hop_depth

        from pragma.query.decomposer import QueryDecomposer
        from pragma.query.retriever import BM25Retriever
        from pragma.query.assembler import FactAssembler
        from pragma.query.synthesizer import AnswerSynthesizer

        if as_of:
            as_of_date = (
                datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
            )
        else:
            as_of_date = None

        decomposer = QueryDecomposer(self._llm, max_subquestions=top_k)
        retriever = BM25Retriever(self._graph_builder)
        assembler = FactAssembler(self._graph_builder, min_confidence=min_confidence)
        synthesizer = AnswerSynthesizer(self._llm)

        sub_questions = decomposer.decompose(query)
        seed_entities = retriever.find_seed_entities(sub_questions)

        if not seed_entities:
            result = PragmaResult(
                answer="Insufficient knowledge in KB for this query",
                reasoning_path=[],
                source_facts=[],
                confidence=0.0,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                subgraph_size=0,
            )
            self._storage.save_query_cache(cache_key, query, result)
            return result

        from pragma.graph.traversal import GraphTraverser

        traverser = GraphTraverser(
            self._graph_builder,
            max_subgraph_nodes=self.config.max_subgraph_nodes,
            default_hop_depth=hop_depth,
        )
        subgraph = traverser.extract_subgraph(seed_entities, hop_depth=hop_depth)

        if subgraph.number_of_nodes() == 0:
            result = PragmaResult(
                answer="Insufficient knowledge in KB for this query",
                reasoning_path=[],
                source_facts=[],
                confidence=0.0,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                subgraph_size=0,
            )
            self._storage.save_query_cache(cache_key, query, result)
            return result

        facts = assembler.assemble_facts(subgraph, as_of=as_of_date)

        if not facts:
            result = PragmaResult(
                answer="Insufficient knowledge in KB for this query",
                reasoning_path=[],
                source_facts=[],
                confidence=0.0,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                subgraph_size=subgraph.number_of_nodes(),
            )
            self._storage.save_query_cache(cache_key, query, result)
            return result

        # Graph reasoning paths used to be sent to the LLM here, but the
        # facts already encode the same structural information and the path
        # text was pure prompt bloat. They are reconstructed downstream from
        # the cited facts when needed.

        # Resolve subject/object UUIDs to entity NAMES so the LLM sees readable
        # references instead of opaque IDs. Without this the model both
        # produces worse answers AND the prompt is bloated with UUID strings.
        ids_to_resolve: set = set()
        for f in facts:
            if f.get("subject_id"):
                ids_to_resolve.add(str(f["subject_id"]))
            if f.get("object_id"):
                ids_to_resolve.add(str(f["object_id"]))
        entity_names: Dict[str, str] = {}
        if ids_to_resolve:
            try:
                conn = self._storage._get_connection()
                placeholders = ",".join("?" * len(ids_to_resolve))
                rows = conn.execute(
                    f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
                    tuple(ids_to_resolve),
                ).fetchall()
                entity_names = {row["id"]: row["name"] for row in rows}
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not resolve entity names: {e}")

        synthesis = synthesizer.synthesize(query, facts, entity_names=entity_names)

        reasoning_steps = [
            ReasoningStep(
                fact_id=step.get("fact_id", ""),
                explanation=step.get("explanation", ""),
                hop_number=i,
            )
            for i, step in enumerate(synthesis.reasoning_steps)
        ]

        source_facts: List[AtomicFact] = []
        for f in facts:
            try:
                source_facts.append(
                    AtomicFact(
                        id=f.get("id", str(uuid.uuid4())),
                        subject_id=f.get("subject_id", ""),
                        predicate=f.get("predicate", ""),
                        object_id=f.get("object_id"),
                        object_value=f.get("object_value"),
                        context=f.get("context", "") or "",
                        source_doc=f.get("source_doc", "") or "",
                        source_page=f.get("source_page"),
                        confidence=float(f.get("confidence", 1.0)),
                        ingested_at=f.get("ingested_at"),
                        is_active=bool(f.get("is_active", True)),
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        # Estimate prompt tokens actually sent to the synthesis LLM. We
        # mirror the synthesizer's pre-filter + compact rendering so the
        # number reflects the real prompt, not pragma's internal fact set.
        # ~4 chars per token is the standard rough conversion for LLMs.
        from pragma.query.synthesizer import (
            DEFAULT_SYNTHESIS_PROMPT,
            AnswerSynthesizer,
        )

        prompt_facts = synthesizer._filter_facts_by_query(facts, query, entity_names)[
            : synthesizer.max_facts
        ]
        fact_chars = sum(
            len(synthesizer._format_fact(f, i + 1, entity_names)) + 1
            for i, f in enumerate(prompt_facts)
        )
        approx_chars = len(DEFAULT_SYNTHESIS_PROMPT) + len(query) + fact_chars + 8
        approx_tokens = max(1, approx_chars // 4)
        del AnswerSynthesizer  # keep import-time symbols out of locals

        result = PragmaResult(
            answer=synthesis.answer,
            reasoning_path=reasoning_steps,
            source_facts=source_facts,
            confidence=synthesis.confidence,
            tokens_used=approx_tokens,
            latency_ms=(time.time() - start_time) * 1000,
            subgraph_size=subgraph.number_of_nodes() + subgraph.number_of_edges(),
        )

        self._storage.save_query_cache(cache_key, query, result)
        return result

    def _compute_query_cache_key(
        self, query: str, hop_depth: int, min_confidence: float, as_of: str
    ) -> str:
        """Compute cache key for query."""
        data = f"{query}:{hop_depth}:{min_confidence}:{as_of}"
        return hashlib.sha256(data.encode()).hexdigest()

    async def stream(
        self,
        query: str,
        hop_depth: Optional[int] = None,
        min_confidence: float = 0.5,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """Stream query answer token by token.

        Steps 1-4 (decompose → retrieve → traverse → assemble) run synchronously.
        Step 5 yields tokens from streaming LLM response.

        Yields:
            str: Token text from LLM

        Example:
            async for token in kb.stream("What is Apple?"):
                print(token, end="", flush=True)
        """

        from pragma.query.decomposer import QueryDecomposer
        from pragma.query.retriever import BM25Retriever
        from pragma.query.assembler import FactAssembler
        from pragma.graph.traversal import GraphTraverser

        if hop_depth is None:
            hop_depth = self.config.default_hop_depth

        decomposer = QueryDecomposer(self._llm, max_subquestions=top_k)
        retriever = BM25Retriever(self._graph_builder)
        assembler = FactAssembler(self._graph_builder, min_confidence=min_confidence)

        sub_questions = decomposer.decompose(query)

        seed_entities = retriever.find_seed_entities(sub_questions)
        if not seed_entities:
            yield "Insufficient knowledge in KB for this query"
            return

        traverser = GraphTraverser(
            self._graph_builder,
            max_subgraph_nodes=self.config.max_subgraph_nodes,
            default_hop_depth=hop_depth,
        )
        subgraph = traverser.extract_subgraph(seed_entities, hop_depth=hop_depth)

        if subgraph.number_of_nodes() == 0:
            yield "Insufficient knowledge in KB for this query"
            return

        facts = assembler.assemble_facts(subgraph)
        if not facts:
            yield "Insufficient knowledge in KB for this query"
            return

        # Resolve entity UUIDs to names so the streamed prompt uses readable
        # references (mirrors the non-streaming path).
        from pragma.query.synthesizer import (
            AnswerSynthesizer,
            DEFAULT_SYNTHESIS_PROMPT,
        )

        ids: set = set()
        for f in facts:
            if f.get("subject_id"):
                ids.add(str(f["subject_id"]))
            if f.get("object_id"):
                ids.add(str(f["object_id"]))
        entity_names: Dict[str, str] = {}
        if ids:
            try:
                conn = self._storage._get_connection()
                placeholders = ",".join("?" * len(ids))
                rows = conn.execute(
                    f"SELECT id, name FROM entities WHERE id IN ({placeholders})",
                    tuple(ids),
                ).fetchall()
                entity_names = {row["id"]: row["name"] for row in rows}
            except Exception as e:  # noqa: BLE001
                logger.debug(f"stream: could not resolve entity names: {e}")

        synth = AnswerSynthesizer(self._llm)
        filtered = synth._filter_facts_by_query(facts, query, entity_names)
        capped = filtered[: synth.max_facts]
        facts_text = "\n".join(
            synth._format_fact(f, i + 1, entity_names) for i, f in enumerate(capped)
        )
        user_prompt = f"{query}\n{facts_text}"

        async for token in self._llm.stream_complete(
            [
                {"role": "system", "content": DEFAULT_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        ):
            yield token

    def close(self) -> None:
        """Close the knowledge base."""
        self._storage.close()
        if hasattr(self._llm, "close"):
            self._llm.close()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class IngestResult:
    """Result of an ingestion operation."""

    def __init__(
        self,
        documents: int = 0,
        facts: int = 0,
        entities: int = 0,
        skipped: int = 0,
    ) -> None:
        self.documents = documents
        self.facts = facts
        self.entities = entities
        self.skipped = skipped

    def __repr__(self) -> str:
        return f"IngestResult(documents={self.documents}, facts={self.facts}, entities={self.entities}, skipped={self.skipped})"

    def summary(self) -> str:
        """Generate summary string."""
        parts = []
        if self.documents:
            parts.append(
                f"{self.documents} document{'s' if self.documents != 1 else ''}"
            )
        if self.facts:
            parts.append(f"{self.facts} fact{'s' if self.facts != 1 else ''}")
        if self.entities:
            parts.append(f"{self.entities} entit{'y' if self.entities == 1 else 'ies'}")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")

        return "Ingested " + ", ".join(parts) if parts else "No changes"
