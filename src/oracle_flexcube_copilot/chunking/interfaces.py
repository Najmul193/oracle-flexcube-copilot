"""Protocol definitions for the chunking pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.enrichment.models import EnrichedDocument


@runtime_checkable
class Chunker(Protocol):
    """Interface for chunking strategies.
    
    Transforms an EnrichedDocument into a list of Chunks.
    """

    def chunk(self, document: EnrichedDocument) -> list[Chunk]:
        """Convert a document into chunks.

        Args:
            document: An EnrichedDocument.

        Returns:
            A list of Chunk instances.
        """
        ...
