"""Tests for chunking interfaces."""

from __future__ import annotations

from oracle_flexcube_copilot.chunking.interfaces import Chunker
from oracle_flexcube_copilot.chunking.strategy import SemanticSectionChunker
from oracle_flexcube_copilot.chunking.models import Chunk


class TestProtocolsAreRuntimeCheckable:
    """All protocols should be decorated with @runtime_checkable."""

    def test_chunker_is_runtime_checkable(self) -> None:
        """Chunker protocol should be runtime-checkable."""
        assert isinstance(SemanticSectionChunker(), Chunker)

    def test_non_conforming_class_fails(self) -> None:
        """A class without the chunk method should not satisfy the protocol."""
        class _NotAChunker:
            def wrong_method(self) -> None:
                pass

        assert not isinstance(_NotAChunker(), Chunker)
