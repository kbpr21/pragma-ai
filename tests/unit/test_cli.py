from typer.testing import CliRunner

from pragma.cli.main import app


runner = CliRunner()


class TestCLI:
    """CLI command tests."""

    def test_cli_help(self):
        """pragma --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Atomic fact reasoning" in result.stdout

    def test_ingest_help(self):
        """pragma ingest --help works."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0

    def test_query_help(self):
        """pragma query --help works."""
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "--hop-depth" in result.stdout

    def test_stats_help(self):
        """pragma stats --help works."""
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_facts_help(self):
        """pragma facts --help works."""
        result = runner.invoke(app, ["facts", "--help"])
        assert result.exit_code == 0

    def test_entities_help(self):
        """pragma entities --help works."""
        result = runner.invoke(app, ["entities", "--help"])
        assert result.exit_code == 0

    def test_config_help(self):
        """pragma config --help works."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_clear_help(self):
        """pragma clear --help works."""
        result = runner.invoke(app, ["clear", "--help"])
        assert result.exit_code == 0

    def test_config_shows_values(self):
        """pragma config shows configuration."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "KB Directory" in result.stdout
        assert "Default Hop Depth" in result.stdout

    def test_clear_aborts_on_no(self, tmp_path):
        """pragma clear aborts on no confirmation."""
        result = runner.invoke(app, ["clear"], input="n\n")
        assert "Aborted" in result.stdout or result.exit_code == 0
