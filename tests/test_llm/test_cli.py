"""Tests for CLI ask command integration with LLM layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from oracle_flexcube_copilot.cli import main


class TestCLIAsk:
    @patch("oracle_flexcube_copilot.cli.EmbeddingCache")
    @patch("oracle_flexcube_copilot.cli.EmbeddingEngine")
    @patch("oracle_flexcube_copilot.cli.VectorRetriever")
    @patch("oracle_flexcube_copilot.cli.BM25Retriever")
    @patch("oracle_flexcube_copilot.cli.RRFFuser")
    @patch("oracle_flexcube_copilot.cli.RAGAnswerGenerator")
    @patch("oracle_flexcube_copilot.cli.RAGPromptBuilder")
    @patch("oracle_flexcube_copilot.cli.ChromaIndexer")
    def test_ask_no_results(
        self,
        _mock_indexer: MagicMock,
        _mock_builder: MagicMock,
        _mock_generator: MagicMock,
        mock_fuser: MagicMock,
        _mock_bm25: MagicMock,
        _mock_vector: MagicMock,
        _mock_embedder: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        mock_fuser.return_value.fuse.return_value = []
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "test query"])
        assert result.exit_code == 0
        assert "No relevant documentation found" in result.output

    @patch("oracle_flexcube_copilot.cli.EmbeddingCache")
    @patch("oracle_flexcube_copilot.cli.EmbeddingEngine")
    @patch("oracle_flexcube_copilot.cli.VectorRetriever")
    @patch("oracle_flexcube_copilot.cli.BM25Retriever")
    @patch("oracle_flexcube_copilot.cli.RRFFuser")
    @patch("oracle_flexcube_copilot.cli.RAGAnswerGenerator")
    @patch("oracle_flexcube_copilot.cli.RAGPromptBuilder")
    @patch("oracle_flexcube_copilot.cli.ConsoleAnswerFormatter")
    @patch("oracle_flexcube_copilot.cli.ChromaIndexer")
    def test_ask_with_results(
        self,
        _mock_indexer: MagicMock,
        mock_formatter: MagicMock,
        mock_builder_cls: MagicMock,
        mock_generator_cls: MagicMock,
        mock_fuser: MagicMock,
        _mock_bm25: MagicMock,
        _mock_vector: MagicMock,
        _mock_embedder: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        mock_fuser.return_value.fuse.return_value = [
            MagicMock(source_document="GL.pdf", page=44, score=0.95, heading="7.1.1")
        ]
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.build.return_value = MagicMock(
            estimated_tokens=100,
            context_blocks=[MagicMock(
                document="GL.pdf", section="7.1.1", page=44, score=0.95, entities=[]
            )],
        )

        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator
        mock_generator.generate.return_value = MagicMock(
            answer="GL Balance Transfer is...",
            citations=[MagicMock(document="GL.pdf", section="7.1.1", page=44, score=0.95)],
            confidence="High",
            confidence_percentage=85.0,
            mode="concise",
            metadata=MagicMock(
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                retrieval_time=0.3, generation_time=2.0, model_name="qwen3:8b",
            ),
        )

        mock_formatter.return_value.format.return_value = "Formatted output"

        runner = CliRunner()
        result = runner.invoke(main, ["ask", "test query", "--no-stream"])
        assert result.exit_code == 0

    @patch("oracle_flexcube_copilot.cli.EmbeddingCache")
    @patch("oracle_flexcube_copilot.cli.EmbeddingEngine")
    @patch("oracle_flexcube_copilot.cli.VectorRetriever")
    @patch("oracle_flexcube_copilot.cli.BM25Retriever")
    @patch("oracle_flexcube_copilot.cli.RRFFuser")
    @patch("oracle_flexcube_copilot.cli.RAGAnswerGenerator")
    @patch("oracle_flexcube_copilot.cli.RAGPromptBuilder")
    @patch("oracle_flexcube_copilot.cli.ConsoleAnswerFormatter")
    @patch("oracle_flexcube_copilot.cli.ChromaIndexer")
    def test_ask_expert_mode(
        self,
        _mock_indexer: MagicMock,
        mock_formatter: MagicMock,
        mock_builder_cls: MagicMock,
        mock_generator_cls: MagicMock,
        mock_fuser: MagicMock,
        _mock_bm25: MagicMock,
        _mock_vector: MagicMock,
        _mock_embedder: MagicMock,
        _mock_cache: MagicMock,
    ) -> None:
        mock_fuser.return_value.fuse.return_value = [
            MagicMock(source_document="GL.pdf", page=44, score=0.95, heading="7.1.1")
        ]
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder
        mock_builder.build.return_value = MagicMock(
            estimated_tokens=100,
            context_blocks=[MagicMock(
                document="GL.pdf", section="7.1.1", page=44, score=0.95, entities=[]
            )],
        )

        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator
        mock_generator.generate.return_value = MagicMock(
            answer="Expert answer...",
            citations=[],
            confidence="High", confidence_percentage=85.0,
            mode="expert",
            metadata=MagicMock(
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                retrieval_time=0.3, generation_time=2.0, model_name="qwen3:8b",
            ),
        )
        mock_formatter.return_value.format.return_value = "Expert output"

        runner = CliRunner()
        result = runner.invoke(main, ["ask", "test query", "--mode", "expert", "--no-stream"])
        assert result.exit_code == 0
