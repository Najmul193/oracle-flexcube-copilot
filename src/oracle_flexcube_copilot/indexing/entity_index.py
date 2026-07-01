"""Oracle Entity exact-match index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.config import settings


class EntityLocation(BaseModel):
    """Represents a specific occurrence of an entity."""

    document_id: str = Field(description="ID of the document")
    document_name: str = Field(description="Name of the document")
    page: int = Field(description="Page number")
    section: str = Field(description="Section or heading")
    chunk_id: str = Field(description="ID of the chunk")


class OracleEntityIndex:
    """Exact lookup index for Oracle entities backed by SQLite."""

    def __init__(self, db_dir: str | None = None):
        """Initialize the entity index SQLite database."""
        db_path_dir = Path(db_dir) if db_dir else settings.resolved_cache_dir.parent / "entity_db"
        db_path_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path_dir / "oracle_entities.sqlite"

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    UNIQUE(entity_name, chunk_id)
                )
                """
            )
            # Create indices for fast lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_locations(entity_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_id ON entity_locations(document_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON entity_locations(chunk_id)")

    def index_chunk(self, chunk: Chunk) -> int:
        """Index all entities found in a chunk."""
        if not chunk.oracle_entities:
            return 0

        doc_name = (
            chunk.metadata.document_name if chunk.metadata and chunk.metadata.document_name else ""
        )
        section = chunk.section_title or ""

        if not section and chunk.heading_path:
            section = chunk.heading_path[-1]

        inserted = 0
        with self._get_connection() as conn:
            for entity in chunk.oracle_entities:
                try:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO entity_locations 
                        (entity_name, entity_type, document_id, document_name, page, section, chunk_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entity.name.upper(),
                            entity.entity_type,
                            chunk.document_id,
                            doc_name,
                            chunk.page_start,
                            section,
                            chunk.id,
                        ),
                    )
                    inserted += cursor.rowcount
                except sqlite3.Error:
                    pass

            conn.commit()

        return inserted

    def lookup(self, entity_name: str) -> list[EntityLocation]:
        """Find all locations where an entity appears (O(1) exact match)."""
        entity_name_upper = entity_name.upper()
        locations = []

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT document_id, document_name, page, section, chunk_id
                FROM entity_locations
                WHERE entity_name = ?
                """,
                (entity_name_upper,),
            )
            for row in cursor:
                locations.append(
                    EntityLocation(
                        document_id=row["document_id"],
                        document_name=row["document_name"],
                        page=row["page"],
                        section=row["section"],
                        chunk_id=row["chunk_id"],
                    )
                )

        return locations

    def remove_document(self, document_id: str) -> int:
        """Remove all entity references for a specific document."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM entity_locations WHERE document_id = ?", (document_id,)
            )
            conn.commit()
            return cursor.rowcount

    def remove_chunk(self, chunk_id: str) -> int:
        """Remove all entity references for a specific chunk."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM entity_locations WHERE chunk_id = ?", (chunk_id,))
            conn.commit()
            return cursor.rowcount

    def clear(self) -> None:
        """Clear the entire index."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM entity_locations")
            conn.commit()
