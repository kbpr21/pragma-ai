import os
import tempfile
from pathlib import Path

import pytest

from pragma.config import PragmaConfig


class TestPragmaConfigDefaults:
    def test_default_values(self):
        config = PragmaConfig()

        assert config.kb_dir == "./pragma_kb"
        assert config.max_facts_per_segment == 50
        assert config.fact_confidence_threshold == 0.6
        assert config.default_hop_depth == 2
        assert config.max_subgraph_nodes == 50
        assert config.max_subquestions == 5
        assert config.enable_query_cache is True
        assert config.query_cache_ttl == 3600
        assert config.fuzzy_match_threshold == 85
        assert config.embeddings_enabled is False
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.llm_provider == "groq"
        assert config.llm_model is None
        assert config.llm_temperature == 0.0
        assert config.max_tokens_per_call == 1000


class TestPragmaConfigFromDict:
    def test_from_dict_with_all_fields(self):
        config_dict = {
            "kb_dir": "./custom_kb",
            "max_facts_per_segment": 50,
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }

        config = PragmaConfig.from_dict(config_dict)

        assert config.kb_dir == "./custom_kb"
        assert config.max_facts_per_segment == 50
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o"

    def test_from_dict_ignores_unknown_fields(self):
        config_dict = {
            "kb_dir": "./custom",
            "unknown_field": "should be ignored",
        }

        config = PragmaConfig.from_dict(config_dict)

        assert config.kb_dir == "./custom"
        assert not hasattr(config, "unknown_field")

    def test_from_dict_with_partial_fields(self):
        config_dict = {"llm_provider": "ollama"}

        config = PragmaConfig.from_dict(config_dict)

        assert config.llm_provider == "ollama"
        assert config.kb_dir == "./pragma_kb"


class TestPragmaConfigFromEnv:
    def test_from_env_no_vars_returns_defaults(self):
        env_vars = [
            "PRAGMA_KB_DIR",
            "PRAGMA_LLM_PROVIDER",
            "PRAGMA_MAX_FACTS_PER_SEGMENT",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        config = PragmaConfig.from_env()

        assert config.kb_dir == "./pragma_kb"
        assert config.llm_provider == "groq"

    def test_from_env_kb_dir(self):
        os.environ["PRAGMA_KB_DIR"] = "/custom/path"

        config = PragmaConfig.from_env()

        assert config.kb_dir == "/custom/path"

        del os.environ["PRAGMA_KB_DIR"]

    def test_from_env_llm_provider(self):
        os.environ["PRAGMA_LLM_PROVIDER"] = "anthropic"

        config = PragmaConfig.from_env()

        assert config.llm_provider == "anthropic"

        del os.environ["PRAGMA_LLM_PROVIDER"]

    def test_from_env_integer_conversion(self):
        os.environ["PRAGMA_MAX_FACTS_PER_SEGMENT"] = "100"
        os.environ["PRAGMA_DEFAULT_HOP_DEPTH"] = "3"
        os.environ["PRAGMA_QUERY_CACHE_TTL"] = "1800"

        config = PragmaConfig.from_env()

        assert config.max_facts_per_segment == 100
        assert config.default_hop_depth == 3
        assert config.query_cache_ttl == 1800

        del os.environ["PRAGMA_MAX_FACTS_PER_SEGMENT"]
        del os.environ["PRAGMA_DEFAULT_HOP_DEPTH"]
        del os.environ["PRAGMA_QUERY_CACHE_TTL"]

    def test_from_env_float_conversion(self):
        os.environ["PRAGMA_FACT_CONFIDENCE_THRESHOLD"] = "0.8"
        os.environ["PRAGMA_LLM_TEMPERATURE"] = "0.5"

        config = PragmaConfig.from_env()

        assert config.fact_confidence_threshold == 0.8
        assert config.llm_temperature == 0.5

        del os.environ["PRAGMA_FACT_CONFIDENCE_THRESHOLD"]
        del os.environ["PRAGMA_LLM_TEMPERATURE"]

    def test_from_env_boolean_conversion(self):
        os.environ["PRAGMA_ENABLE_QUERY_CACHE"] = "false"
        os.environ["PRAGMA_EMBEDDINGS_ENABLED"] = "true"

        config = PragmaConfig.from_env()

        assert config.enable_query_cache is False
        assert config.embeddings_enabled is True

        del os.environ["PRAGMA_ENABLE_QUERY_CACHE"]
        del os.environ["PRAGMA_EMBEDDINGS_ENABLED"]

    def test_from_env_boolean_true_values(self):
        for value in ("true", "1", "yes"):
            os.environ["PRAGMA_ENABLE_QUERY_CACHE"] = value
            config = PragmaConfig.from_env()
            assert config.enable_query_cache is True
            del os.environ["PRAGMA_ENABLE_QUERY_CACHE"]


class TestPragmaConfigFromYaml:
    def test_from_yaml_file_not_found(self):
        pytest.importorskip("yaml")

        with pytest.raises(FileNotFoundError):
            PragmaConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file_returns_defaults(self):
        pytest.importorskip("yaml")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name

        try:
            config = PragmaConfig.from_yaml(path)
            assert config.kb_dir == "./pragma_kb"
        finally:
            Path(path).unlink()

    def test_from_yaml_with_pragma_section(self):
        pytest.importorskip("yaml")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("pragma:\n  kb_dir: ./from_yaml\n  llm_provider: openai\n")
            path = f.name

        try:
            config = PragmaConfig.from_yaml(path)
            assert config.kb_dir == "./from_yaml"
            assert config.llm_provider == "openai"
        finally:
            Path(path).unlink()

    def test_from_yaml_without_pragma_section(self):
        pytest.importorskip("yaml")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("other_section:\n  value: test\n")
            path = f.name

        try:
            config = PragmaConfig.from_yaml(path)
            assert config.kb_dir == "./pragma_kb"
        finally:
            Path(path).unlink()


class TestPragmaConfigToDict:
    def test_to_dict_returns_all_fields(self):
        config = PragmaConfig(kb_dir="./test", llm_provider="ollama")

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["kb_dir"] == "./test"
        assert config_dict["llm_provider"] == "ollama"
        assert "max_facts_per_segment" in config_dict


class TestPragmaConfigValidation:
    def test_max_subgraph_nodes_floor(self):
        # Values below 5 are clamped up to 5 to keep multi-hop recall
        # non-trivial; values >= 5 are passed through unchanged.
        assert PragmaConfig(max_subgraph_nodes=3).max_subgraph_nodes == 5
        assert PragmaConfig(max_subgraph_nodes=0).max_subgraph_nodes == 5
        assert PragmaConfig(max_subgraph_nodes=10).max_subgraph_nodes == 10
        assert PragmaConfig(max_subgraph_nodes=50).max_subgraph_nodes == 50
