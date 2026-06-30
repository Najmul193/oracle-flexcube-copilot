"""Tests for the CLI."""

from unittest.mock import patch
from click.testing import CliRunner

from oracle_flexcube_copilot.cli import main


def test_cli_stats() -> None:
    """It should execute the stats command successfully."""
    runner = CliRunner()
    
    with patch("oracle_flexcube_copilot.cli.ChromaIndexer") as MockIndexer:
        # Mock indexer health
        mock_instance = MockIndexer.return_value
        mock_instance.health_check.return_value.is_accessible = True
        mock_instance.health_check.return_value.total_chunks = 42
        mock_instance.db_dir = "/tmp/mock_db"
        
        with patch("oracle_flexcube_copilot.cli.OracleEntityIndex"):
            result = runner.invoke(main, ["stats"])
            
            assert result.exit_code == 0
            assert "System Health Statistics" in result.output
            assert "ChromaDB Vector Index" in result.output
            assert "SQLite Entity Index" in result.output


def test_cli_search() -> None:
    """It should execute the search command and print results."""
    runner = CliRunner()
    
    with patch("oracle_flexcube_copilot.cli.VectorRetriever") as MockRetriever:
        mock_instance = MockRetriever.return_value
        # Use a real list, but we can mock the inner SearchResult if we want or just mock empty
        mock_instance.retrieve.return_value = []
        
        with patch("oracle_flexcube_copilot.cli.ChromaIndexer"):
            with patch("oracle_flexcube_copilot.cli.EmbeddingEngine"):
                result = runner.invoke(main, ["search", "test query"])
                
                assert result.exit_code == 0
                assert "Searching for" in result.output
                assert "No results found" in result.output
