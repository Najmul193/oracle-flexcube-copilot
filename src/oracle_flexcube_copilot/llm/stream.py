"""Stream handler for LLM token streaming."""

from __future__ import annotations

import logging
from collections.abc import Iterator

logger = logging.getLogger("oracle_flexcube_copilot.llm.stream")


class StreamHandler:
    """Handles streaming tokens from the generator and tracks statistics."""

    def __init__(self) -> None:
        self.token_count = 0
        self.full_text: list[str] = []

    def handle(self, stream: Iterator[str]) -> Iterator[str]:
        """Process a token stream, tracking stats.

        Args:
            stream: Token iterator from the generator.

        Yields:
            Each token with formatting preserved.
        """
        self.token_count = 0
        self.full_text = []

        for token in stream:
            self.token_count += 1
            self.full_text.append(token)
            yield token

    @property
    def text(self) -> str:
        """Return the complete accumulated text."""
        return "".join(self.full_text)
