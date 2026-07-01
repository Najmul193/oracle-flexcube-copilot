"""Console formatter for AnswerResponse."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.models import AnswerResponse


class ConsoleAnswerFormatter:
    """Formats AnswerResponse for CLI display."""

    SEPARATOR = "\n" + "-" * 60

    def format(self, response: AnswerResponse, question: str) -> str:
        """Format a complete answer response for console display.

        Args:
            response: The answer response to format.
            question: The original user question.

        Returns:
            A formatted string for console output.
        """
        parts: list[str] = []

        parts.append(f"Question: {question}")
        parts.append("")
        parts.append(f"Answer [{response.mode}]")
        parts.append(response.answer)

        if response.citations:
            parts.append("")
            parts.append("Sources")
            for c in response.citations:
                section_str = f" | Section: {c.section}" if c.section else ""
                parts.append(f"  {c.document} (Page {c.page}){section_str}")

        parts.append("")
        confidence_str = (
            f"Confidence: {response.confidence} ({response.confidence_percentage:.0f}%)"
        )
        parts.append(confidence_str)

        meta = response.metadata
        retrieval_ms = meta.retrieval_time * 1000
        gen_ms = meta.generation_time * 1000
        total_ms = (meta.retrieval_time + meta.generation_time) * 1000
        parts.append(
            f"Retrieval: {retrieval_ms:.0f}ms | "
            f"Generation: {gen_ms:.0f}ms | "
            f"Total: {total_ms:.0f}ms"
        )
        parts.append(
            f"Tokens: {meta.prompt_tokens} prompt + {meta.completion_tokens} completion = {meta.total_tokens} total"
        )
        parts.append(f"Model: {meta.model_name}")

        parts.append(self.SEPARATOR)

        return "\n".join(parts)

    def format_stream_start(self, question: str, mode: str) -> str:
        """Format the header shown before streaming begins."""
        return f"Question: {question}\n\nAnswer [{mode}]:\n"

    def format_stream_end(self, response: AnswerResponse) -> str:
        """Format the footer shown after streaming completes."""
        parts: list[str] = []

        if response.citations:
            parts.append("")
            parts.append("Sources")
            for c in response.citations:
                section_str = f" | Section: {c.section}" if c.section else ""
                parts.append(f"  {c.document} (Page {c.page}){section_str}")

        confidence_str = (
            f"Confidence: {response.confidence} ({response.confidence_percentage:.0f}%)"
        )
        parts.append("")
        parts.append(confidence_str)

        meta = response.metadata
        retrieval_ms = meta.retrieval_time * 1000
        gen_ms = meta.generation_time * 1000
        total_ms = (meta.retrieval_time + meta.generation_time) * 1000
        parts.append(
            f"Retrieval: {retrieval_ms:.0f}ms | "
            f"Generation: {gen_ms:.0f}ms | "
            f"Total: {total_ms:.0f}ms"
        )
        parts.append(
            f"Tokens: {meta.prompt_tokens} prompt + {meta.completion_tokens} completion = {meta.total_tokens} total"
        )
        parts.append(f"Model: {meta.model_name}")

        parts.append(ConsoleAnswerFormatter.SEPARATOR)

        return "\n".join(parts)
