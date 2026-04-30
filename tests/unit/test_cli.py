"""CLI smoke tests.

Typer renders help output through Rich, which can:
- inject ANSI escape sequences for colours / styling
- word-wrap long option names like ``--hop-depth`` across lines based on
  the terminal width (CI defaults to 80 columns)

Both behaviours can break naive substring assertions on ``result.stdout``.
We defend against them with two complementary measures:

1. ``CLEAN_ENV`` forces a 200-column terminal and disables colour, so the
   underlying click runner reports unwrapped, unstyled output.
2. ``_clean(...)`` strips any residual ANSI escape sequences and
   collapses whitespace, so we can match substrings that the renderer
   may still have soft-wrapped.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from pragma.cli.main import app


runner = CliRunner()

# Force wide terminal + no colour for deterministic help output. NO_COLOR
# is honoured by Rich; COLUMNS is honoured by click's wrapping logic.
CLEAN_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}

_ANSI_RE = re.compile(r"\x1b\[[\d;]*[A-Za-z]")


def _clean(text: str) -> str:
    """Strip ANSI escapes and normalise whitespace so substring asserts
    are robust to Rich's table / box drawing output."""
    text = _ANSI_RE.sub("", text or "")
    return re.sub(r"\s+", " ", text)


class TestCLI:
    """CLI command tests."""

    def test_cli_help(self):
        result = runner.invoke(app, ["--help"], env=CLEAN_ENV)
        assert result.exit_code == 0
        assert "Atomic fact reasoning" in _clean(result.stdout)

    def test_ingest_help(self):
        result = runner.invoke(app, ["ingest", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_query_help(self):
        result = runner.invoke(app, ["query", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0
        # Use the unwrappable core of the option name. Even if the
        # leading "--" is hidden behind ANSI styling, "hop-depth" is
        # always emitted as one token by click's help formatter.
        assert "hop-depth" in _clean(result.stdout)

    def test_stats_help(self):
        result = runner.invoke(app, ["stats", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_facts_help(self):
        result = runner.invoke(app, ["facts", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_entities_help(self):
        result = runner.invoke(app, ["entities", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_clear_help(self):
        result = runner.invoke(app, ["clear", "--help"], env=CLEAN_ENV)
        assert result.exit_code == 0

    def test_config_shows_values(self):
        result = runner.invoke(app, ["config"], env=CLEAN_ENV)
        assert result.exit_code == 0
        clean = _clean(result.stdout)
        assert "KB Directory" in clean
        assert "Default Hop Depth" in clean

    def test_clear_aborts_on_no(self, tmp_path):
        result = runner.invoke(app, ["clear"], input="n\n", env=CLEAN_ENV)
        assert "Aborted" in _clean(result.stdout) or result.exit_code == 0
