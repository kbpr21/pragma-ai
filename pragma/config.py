from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import os


@dataclass
class PragmaConfig:
    """Configuration for pragma KnowledgeBase."""

    kb_dir: str = "./pragma_kb"

    max_facts_per_segment: int = 50
    fact_confidence_threshold: float = 0.6

    default_hop_depth: int = 2
    # 50 was the original design target. v1.0.1 shipped with 5 by
    # mistake, which silently capped multi-hop recall on any KB beyond
    # toy size: a 2-hop traversal that finds 6 candidate nodes would
    # drop the 6th. v1.0.2 restores 50; users who really want a tighter
    # context window can set PRAGMA_MAX_SUBGRAPH_NODES.
    max_subgraph_nodes: int = 50

    max_subquestions: int = 5
    enable_query_cache: bool = True
    query_cache_ttl: int = 3600

    fuzzy_match_threshold: int = 85

    embeddings_enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"

    llm_provider: str = "groq"
    llm_model: Optional[str] = None
    llm_temperature: float = 0.0
    max_tokens_per_call: int = 1000

    def __post_init__(self) -> None:
        # Floor of 5 keeps obviously-wrong values (0, negative) from
        # producing empty subgraphs. We do NOT raise above 5 anymore --
        # users who explicitly set a small value (e.g. for unit tests
        # or memory-constrained edge devices) are trusted.
        if self.max_subgraph_nodes < 5:
            self.max_subgraph_nodes = 5

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PragmaConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_env(cls) -> "PragmaConfig":
        env_mappings = {
            "PRAGMA_KB_DIR": "kb_dir",
            "PRAGMA_MAX_FACTS_PER_SEGMENT": "max_facts_per_segment",
            "PRAGMA_FACT_CONFIDENCE_THRESHOLD": "fact_confidence_threshold",
            "PRAGMA_DEFAULT_HOP_DEPTH": "default_hop_depth",
            "PRAGMA_MAX_SUBGRAPH_NODES": "max_subgraph_nodes",
            "PRAGMA_MAX_SUBQUESTIONS": "max_subquestions",
            "PRAGMA_ENABLE_QUERY_CACHE": "enable_query_cache",
            "PRAGMA_QUERY_CACHE_TTL": "query_cache_ttl",
            "PRAGMA_FUZZY_MATCH_THRESHOLD": "fuzzy_match_threshold",
            "PRAGMA_EMBEDDINGS_ENABLED": "embeddings_enabled",
            "PRAGMA_EMBEDDING_MODEL": "embedding_model",
            "PRAGMA_LLM_PROVIDER": "llm_provider",
            "PRAGMA_LLM_MODEL": "llm_model",
            "PRAGMA_LLM_TEMPERATURE": "llm_temperature",
            "PRAGMA_MAX_TOKENS_PER_CALL": "max_tokens_per_call",
        }

        config: dict[str, Any] = {}
        for env_var, config_field in env_mappings.items():
            if env_var in os.environ:
                value: Any = os.environ[env_var]
                if config_field in (
                    "max_facts_per_segment",
                    "default_hop_depth",
                    "max_subgraph_nodes",
                    "max_subquestions",
                    "query_cache_ttl",
                    "fuzzy_match_threshold",
                    "max_tokens_per_call",
                ):
                    value = int(value)
                elif config_field in ("fact_confidence_threshold", "llm_temperature"):
                    value = float(value)
                elif config_field == "enable_query_cache":
                    value = value.lower() in ("true", "1", "yes")
                elif config_field == "embeddings_enabled":
                    value = value.lower() in ("true", "1", "yes")
                config[config_field] = value

        return cls.from_dict(config)

    @classmethod
    def default(cls) -> "PragmaConfig":
        """Get default configuration."""
        return cls(
            kb_dir=os.environ.get("PRAGMA_KB_DIR", "./pragma_kb"),
            max_facts_per_segment=30,
            fact_confidence_threshold=0.6,
            default_hop_depth=2,
            max_subgraph_nodes=50,
            max_subquestions=5,
            enable_query_cache=True,
            query_cache_ttl=3600,
            fuzzy_match_threshold=85,
            embeddings_enabled=False,
            embedding_model="all-MiniLM-L6-v2",
            llm_provider="inception",
            llm_model=None,
            llm_temperature=0.0,
            max_tokens_per_call=1000,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PragmaConfig":
        try:
            import yaml
        except ImportError:
            msg = "PyYAML is required for YAML config loading. Install with: pip install pyyaml"
            raise ImportError(msg)

        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        config_dict: dict[str, Any] = {}
        if "pragma" in data:
            config_dict = data["pragma"]

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_dir": self.kb_dir,
            "max_facts_per_segment": self.max_facts_per_segment,
            "fact_confidence_threshold": self.fact_confidence_threshold,
            "default_hop_depth": self.default_hop_depth,
            "max_subgraph_nodes": self.max_subgraph_nodes,
            "max_subquestions": self.max_subquestions,
            "enable_query_cache": self.enable_query_cache,
            "query_cache_ttl": self.query_cache_ttl,
            "fuzzy_match_threshold": self.fuzzy_match_threshold,
            "embeddings_enabled": self.embeddings_enabled,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "max_tokens_per_call": self.max_tokens_per_call,
        }
