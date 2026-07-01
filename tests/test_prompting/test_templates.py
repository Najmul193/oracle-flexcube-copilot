from __future__ import annotations

from oracle_flexcube_copilot.prompting.models import ContextBlock
from oracle_flexcube_copilot.prompting.templates import (
    SystemPromptBuilder,
    render_context_block,
    render_context_xml,
    render_date,
)


def test_system_prompt_strict_mode() -> None:
    builder = SystemPromptBuilder()
    prompt = builder.build(mode="strict")
    assert "Oracle FLEXCUBE Copilot" in prompt
    assert "Answering Policy" in prompt
    assert "Use only the supplied Oracle documentation" in prompt


def test_system_prompt_benchmark_mode() -> None:
    builder = SystemPromptBuilder()
    prompt = builder.build(mode="benchmark")
    assert "Oracle FLEXCUBE Copilot" in prompt
    assert "Answering Policy" not in prompt


def test_system_prompt_support_mode() -> None:
    builder = SystemPromptBuilder()
    prompt = builder.build(mode="support")
    assert "Oracle FLEXCUBE Copilot" in prompt
    assert "Answering Policy" in prompt


def test_render_date() -> None:
    rendered = render_date()
    assert len(rendered) == 10  # YYYY-MM-DD
    assert rendered.count("-") == 2


def test_render_context_block() -> None:
    block = ContextBlock(
        chunk_id="c1",
        document="GL.pdf",
        document_id="abc123",
        section="GL Transfer",
        page=42,
        score=0.9,
        entities=["STTM_GL"],
        text="This is the content.",
        index=1,
    )
    xml = render_context_block(block)
    assert '<context_block index="1"' in xml
    assert 'id="c1"' in xml
    assert "<chunk_id>c1</chunk_id>" in xml
    assert "<document>GL.pdf</document>" in xml
    assert "<document_id>abc123</document_id>" in xml
    assert "<section>GL Transfer</section>" in xml
    assert "<page>42</page>" in xml
    assert "<entities>STTM_GL</entities>" in xml
    assert "This is the content." in xml
    assert "</context_block>" in xml


def test_render_context_block_no_section() -> None:
    block = ContextBlock(
        chunk_id="c1",
        document="doc.pdf",
        page=1,
        score=0.5,
        text="no section",
        index=2,
    )
    xml = render_context_block(block)
    assert "<section></section>" in xml
    assert "<section_id></section_id>" in xml
    assert "<document_id></document_id>" in xml
    assert "<module></module>" in xml
    assert "<entities></entities>" in xml


def test_render_context_xml() -> None:
    blocks = [
        ContextBlock(chunk_id="c1", document="a.pdf", page=1, score=0.9, text="a", index=1),
        ContextBlock(chunk_id="c2", document="b.pdf", page=2, score=0.8, text="b", index=2),
    ]
    xml = render_context_xml(blocks)
    assert xml.startswith("<context>")
    assert xml.endswith("</context>")
    assert xml.count("<context_block") == 2


def test_render_context_xml_empty() -> None:
    xml = render_context_xml([])
    assert xml == "<context>\n</context>"


def test_xml_escaping() -> None:
    block = ContextBlock(
        chunk_id="c1",
        document="test.pdf",
        page=1,
        score=0.5,
        text="A & B < C > D \"quote\" 'single'",
        index=1,
    )
    xml = render_context_block(block)
    assert "A &amp; B" in xml
    assert "&lt; C &gt; D" in xml
    assert "&quot;quote&quot;" in xml
    assert "&apos;single&apos;" in xml


def test_render_date_included() -> None:
    date_str = render_date()
    parts = date_str.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4  # year
    assert 1 <= int(parts[1]) <= 12  # month
    assert 1 <= int(parts[2]) <= 31  # day
