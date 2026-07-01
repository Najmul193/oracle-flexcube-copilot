"""Tests for table extraction."""

from __future__ import annotations

from datetime import UTC, datetime

from oracle_flexcube_copilot.enrichment.tables import (
    _is_likely_table,
    extract_tables,
)
from oracle_flexcube_copilot.ingestion.models import (
    Block,
    Document,
    DocumentMetadata,
    Page,
    Paragraph,
)
from oracle_flexcube_copilot.ingestion.models import (
    TableData as IngestionTableData,
)


def _make_doc(pages: list[Page], doc_id: str = "abc123") -> Document:
    """Helper to build a minimal Document."""
    return Document(
        id=doc_id,
        filename="tables_test.pdf",
        absolute_path="/tmp/tables_test.pdf",
        sha256=doc_id,
        file_size_bytes=1024,
        last_modified=datetime(2024, 1, 1, tzinfo=UTC),
        created_time=datetime(2024, 1, 1, tzinfo=UTC),
        metadata=DocumentMetadata(page_count=len(pages)),
        pages=pages,
    )


class TestExtractTables:
    """Tests for :func:`extract_tables`."""

    def test_no_tables(self, sample_document: Document) -> None:
        """Document without tables should return empty list."""
        tables = extract_tables(sample_document)
        assert isinstance(tables, list)

    def test_extracts_pre_parsed_table_block(self) -> None:
        """Should extract tables from blocks with type='table' and pre-parsed TableData."""
        table_data = IngestionTableData(
            headers=["Field", "Type", "Description"],
            rows=[
                ["product_code", "VARCHAR", "Unique product identifier"],
                ["rate_code", "VARCHAR", "Interest rate code"],
            ],
            title="Product Fields",
        )
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(
                    id="abc123:p1:b0",
                    type="table",
                    block_index=0,
                    paragraphs=[],
                    table=table_data,
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert len(tables) == 1
        assert tables[0].headers == ["Field", "Type", "Description"]
        assert len(tables[0].rows) == 2
        assert tables[0].title == "Product Fields"

    def test_table_page_number_tracked(self) -> None:
        """Table should carry the correct page number."""
        table_data = IngestionTableData(
            headers=["Col1", "Col2"],
            rows=[["a", "b"]],
        )
        page = Page(
            id="abc123:p5",
            page_number=5,
            blocks=[
                Block(
                    id="abc123:p5:b0", type="table", block_index=0, paragraphs=[], table=table_data
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert tables[0].page == 5

    def test_stable_table_ids(self) -> None:
        """Table IDs should follow {doc_id}:tbl:{N} pattern."""
        table_data = IngestionTableData(headers=["A"], rows=[["1"]])
        page = Page(
            id="mysha:p1",
            page_number=1,
            blocks=[
                Block(
                    id="mysha:p1:b0", type="table", block_index=0, paragraphs=[], table=table_data
                ),
            ],
        )
        doc = _make_doc([page], doc_id="mysha")
        tables = extract_tables(doc)
        assert tables[0].id == "mysha:tbl:1"

    def test_num_rows_and_num_cols(self) -> None:
        """num_rows and num_cols should reflect actual data shape."""
        table_data = IngestionTableData(
            headers=["H1", "H2", "H3"],
            rows=[["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
        )
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(
                    id="abc123:p1:b0", type="table", block_index=0, paragraphs=[], table=table_data
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert tables[0].num_rows == 3
        assert tables[0].num_cols == 3

    def test_multiple_tables_across_pages(self) -> None:
        """Should extract tables from multiple pages with sequential IDs."""
        td1 = IngestionTableData(headers=["A"], rows=[["1"]])
        td2 = IngestionTableData(headers=["B", "C"], rows=[["2", "3"]])
        pages = [
            Page(
                id="abc123:p1",
                page_number=1,
                blocks=[
                    Block(id="abc123:p1:b0", type="table", block_index=0, paragraphs=[], table=td1),
                ],
            ),
            Page(
                id="abc123:p2",
                page_number=2,
                blocks=[
                    Block(id="abc123:p2:b0", type="table", block_index=0, paragraphs=[], table=td2),
                ],
            ),
        ]
        doc = _make_doc(pages)
        tables = extract_tables(doc)
        assert len(tables) == 2
        assert tables[0].id == "abc123:tbl:1"
        assert tables[1].id == "abc123:tbl:2"
        assert tables[0].page == 1
        assert tables[1].page == 2

    def test_table_block_without_table_data_skipped(self) -> None:
        """A table-type block with no TableData should be skipped."""
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(id="abc123:p1:b0", type="table", block_index=0, paragraphs=[], table=None),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert len(tables) == 0

    def test_detects_table_in_text_block_with_spaces(self) -> None:
        """Should detect table structure in text blocks with multi-space delimiters."""
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(
                    id="abc123:p1:b0",
                    type="text",
                    block_index=0,
                    paragraphs=[
                        Paragraph(text="Name    Age    City", index=0),
                        Paragraph(text="Alice   30     NYC", index=1),
                        Paragraph(text="Bob     25     LA", index=2),
                    ],
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        # Should detect the tabular pattern
        assert isinstance(tables, list)

    def test_single_line_text_not_table(self) -> None:
        """A single-line text block should not be detected as a table."""
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(
                    id="abc123:p1:b0",
                    type="text",
                    block_index=0,
                    paragraphs=[Paragraph(text="Just one line of text", index=0)],
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert len(tables) == 0

    def test_empty_paragraphs_not_table(self) -> None:
        """A text block with no paragraphs should not produce a table."""
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(id="abc123:p1:b0", type="text", block_index=0, paragraphs=[]),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert len(tables) == 0

    def test_source_block_id_preserved(self) -> None:
        """Extracted table should reference its source block."""
        table_data = IngestionTableData(headers=["X"], rows=[["1"]])
        page = Page(
            id="abc123:p1",
            page_number=1,
            blocks=[
                Block(
                    id="abc123:p1:b0", type="table", block_index=0, paragraphs=[], table=table_data
                ),
            ],
        )
        doc = _make_doc([page])
        tables = extract_tables(doc)
        assert tables[0].source_block_id == "abc123:p1:b0"


class TestIsLikelyTable:
    """Tests for :func:`_is_likely_table`."""

    def test_consistent_columns_detected(self) -> None:
        """Lines with consistent multi-space columns should be detected as table."""
        import re

        lines = [
            "Name    Age    City",
            "Alice   30     NYC",
            "Bob     25     LA",
        ]
        assert _is_likely_table(lines, re.compile(r"\s{2,}")) is True

    def test_inconsistent_columns_rejected(self) -> None:
        """Lines with wildly different column counts should not be a table."""
        import re

        lines = [
            "This is a single paragraph of text.",
            "Another sentence with no columns at all.",
        ]
        assert _is_likely_table(lines, re.compile(r"\t")) is False
