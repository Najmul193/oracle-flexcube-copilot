"""Semantic Section Chunker strategy for Module 4."""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from oracle_flexcube_copilot.chunking.interfaces import Chunker
from oracle_flexcube_copilot.chunking.models import Chunk, ChunkMetadata
from oracle_flexcube_copilot.enrichment.models import EnrichedDocument
from oracle_flexcube_copilot.ingestion.models import Block
from oracle_flexcube_copilot.config import settings

logger = logging.getLogger("oracle_flexcube_copilot.chunking.strategy")

# Procedure step detection pattern
PROCEDURE_PATTERN = re.compile(r"^(step\s*\d+|[a-z]\)|\d+\.)", re.IGNORECASE)


class SemanticSectionChunker(Chunker):
    """Chunks documents semantically based on section boundaries and unbreakable blocks."""

    def __init__(self, target_tokens: int = 800, max_tokens: int = 900, overlap_tokens: int = 100):
        """Initialize the chunker with size constraints.
        
        Args:
            target_tokens: Desired target size in tokens.
            max_tokens: Hard maximum size in tokens.
            overlap_tokens: Number of overlapping tokens between chunks in same section.
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        self.pipeline_meta = ChunkMetadata(
            pipeline_version=settings.pipeline_version,
            chunking_version=settings.chunking_version,
            embedding_model=settings.embedding_model,
            embedding_version=settings.embedding_version,
        )

    def chunk(self, document: EnrichedDocument) -> list[Chunk]:
        """Convert a document into semantic chunks."""
        logger.info("Chunking document: %s", document.filename)
        chunks: list[Chunk] = []
        chunk_counter = 0

        # Create block lookup map
        block_map: dict[str, Block] = {}
        page_map: dict[str, int] = {}
        for page in document.pages:
            for block in page.blocks:
                block_map[block.id] = block
                page_map[block.id] = page.page_number

        # Map blocks to heading paths and section info
        enriched_block_map = {eb.id: eb for eb in document.enriched_blocks}

        for section in document.sections:
            if not section.block_ids:
                continue

            # Accumulate text for this section
            current_chunk_text = ""
            current_chunk_word_count = 0
            current_blocks: list[Block] = []
            
            for block_id in section.block_ids:
                block = block_map.get(block_id)
                if not block:
                    continue
                
                # Check if block is unbreakable
                is_unbreakable = self._is_unbreakable(block)
                block_text = self._get_block_text(block)
                block_word_count = len(block_text.split())
                
                # Token heuristic
                block_token_count = int(block_word_count * 1.3)
                current_token_count = int(current_chunk_word_count * 1.3)

                if current_token_count + block_token_count > self.max_tokens and current_chunk_word_count > 0:
                    # Flush current chunk
                    chunk_counter += 1
                    chunks.append(self._create_chunk(
                        document=document,
                        chunk_id=f"{document.document_id}:chunk:{chunk_counter}",
                        text=current_chunk_text,
                        word_count=current_chunk_word_count,
                        blocks=current_blocks,
                        section=section,
                        enriched_block_map=enriched_block_map,
                        page_map=page_map,
                    ))
                    
                    # Overlap logic: keep last few paragraphs
                    overlap_text, overlap_words, overlap_blocks = self._get_overlap(
                        current_blocks, self.overlap_tokens
                    )
                    
                    current_chunk_text = overlap_text + "\n\n" + block_text
                    current_chunk_word_count = overlap_words + block_word_count
                    current_blocks = overlap_blocks + [block]
                else:
                    if current_chunk_text:
                        current_chunk_text += "\n\n"
                    current_chunk_text += block_text
                    current_chunk_word_count += block_word_count
                    current_blocks.append(block)

            # Flush remaining for section
            if current_chunk_word_count > 0:
                chunk_counter += 1
                chunks.append(self._create_chunk(
                    document=document,
                    chunk_id=f"{document.document_id}:chunk:{chunk_counter}",
                    text=current_chunk_text,
                    word_count=current_chunk_word_count,
                    blocks=current_blocks,
                    section=section,
                    enriched_block_map=enriched_block_map,
                    page_map=page_map,
                ))

        logger.info("Generated %d chunks", len(chunks))
        return chunks

    def _is_unbreakable(self, block: Block) -> bool:
        """Check if a block should not be split (tables, code, lists, procedures)."""
        if block.type in ("table", "list", "code"):
            return True
            
        # Check if first paragraph looks like a procedural step
        if block.paragraphs:
            first_text = block.paragraphs[0].text
            if PROCEDURE_PATTERN.match(first_text):
                return True
                
        return False

    def _get_block_text(self, block: Block) -> str:
        """Get full text for a block."""
        if block.type == "table" and block.table:
            # Simple markdown representation for tables
            lines = []
            if block.table.title:
                lines.append(f"**{block.table.title}**")
            
            if block.table.headers:
                lines.append(" | ".join(block.table.headers))
                lines.append(" | ".join(["---"] * len(block.table.headers)))
                
            for row in block.table.rows:
                lines.append(" | ".join(row))
                
            return "\n".join(lines)
            
        return "\n\n".join(p.text for p in block.paragraphs)

    def _get_overlap(self, blocks: list[Block], target_overlap_tokens: int) -> tuple[str, int, list[Block]]:
        """Extract overlapping text from the end of the current blocks.
        
        Attempts to respect unbreakable boundaries.
        """
        if not blocks:
            return "", 0, []
            
        overlap_blocks: list[Block] = []
        overlap_words = 0
        
        # Traverse backwards
        for block in reversed(blocks):
            block_words = len(self._get_block_text(block).split())
            if (overlap_words + block_words) * 1.3 > target_overlap_tokens:
                break
            
            overlap_blocks.insert(0, block)
            overlap_words += block_words
            
        overlap_text = "\n\n".join(self._get_block_text(b) for b in overlap_blocks)
        return overlap_text, overlap_words, overlap_blocks

    def _create_chunk(
        self,
        document: EnrichedDocument,
        chunk_id: str,
        text: str,
        word_count: int,
        blocks: list[Block],
        section: Any,
        enriched_block_map: dict[str, Any],
        page_map: dict[str, int],
    ) -> Chunk:
        """Create a Chunk object populated with all metadata."""
        # Determine page range
        pages = [page_map[b.id] for b in blocks if b.id in page_map]
        page_start = min(pages) if pages else 0
        page_end = max(pages) if pages else 0
        
        # Determine heading path (use the last block's path to represent current context)
        heading_path = []
        if blocks:
            eb = enriched_block_map.get(blocks[-1].id)
            if eb:
                heading_path = eb.heading_path

        # Collect entities, references, tables
        block_ids = {b.id for b in blocks}
        
        # Filter entities intersecting with this chunk
        entities = [e for e in document.oracle_entities if e.section_id == section.id and e.page in pages]
        
        # Filter references matching block IDs
        references = [r for r in document.cross_references if r.source_block_id in block_ids]
        
        # Filter tables matching block IDs
        table_ids = [t.id for t in document.tables if t.source_block_id in block_ids]

        return Chunk(
            id=chunk_id,
            document_id=document.document_id,
            text=text.strip(),
            heading_path=heading_path,
            section_title=section.title,
            page_start=page_start,
            page_end=page_end,
            oracle_entities=entities,
            references=references,
            table_ids=table_ids,
            token_count=int(word_count * 1.3),
            word_count=word_count,
            metadata=self.pipeline_meta,
        )
