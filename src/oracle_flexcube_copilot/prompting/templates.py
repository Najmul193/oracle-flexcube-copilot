from __future__ import annotations

from datetime import date

from oracle_flexcube_copilot.prompting.models import ContextBlock

_ANSWER_POLICY = """Answering Policy
1. Use only the supplied Oracle documentation.
2. If the answer is incomplete, explicitly state that the documentation provided is insufficient.
3. Never invent menu names, screen names, parameters, or procedures.
4. Quote Oracle terminology exactly when possible.
5. Always include citations in the format: (Document, Section, Page)
6. If multiple documents disagree, mention the discrepancy instead of choosing one.
7. If the user's question is ambiguous, explain the ambiguity and answer based on the available context."""

_SYSTEM_INTRO = (
    "You are an Oracle FLEXCUBE Copilot \u2014 a specialized assistant "
    "for Oracle FLEXCUBE documentation."
)


class SystemPromptBuilder:
    """Builds system prompts for different interaction modes.

    Modes:
        strict:    Full answering policy, citation enforcement (default).
        support:   Softer tone, still citation-bound.
        benchmark: Minimized prompt for evaluation runs.
    """

    def build(self, mode: str = "strict") -> str:
        """Return a system prompt string for the given *mode*."""
        if mode == "benchmark":
            return _SYSTEM_INTRO
        if mode == "support":
            return f"{_SYSTEM_INTRO}\n\n{_ANSWER_POLICY}"
        return f"{_SYSTEM_INTRO}\n\n{_ANSWER_POLICY}"


def render_date() -> str:
    """Return today's date string for inclusion in prompts."""
    return date.today().isoformat()


CONTEXT_BLOCK_XML = """<context_block index=\"{index}\" id=\"{chunk_id}\">
<chunk_id>{chunk_id}</chunk_id>
<document>{document}</document>
<document_id>{document_id}</document_id>
<section>{section}</section>
<section_id>{section_id}</section_id>
<module>{module}</module>
<page>{page}</page>
<page_end>{page_end}</page_end>
<entities>{entities}</entities>
<text>
{text}
</text>
</context_block>"""


def render_context_block(block: ContextBlock) -> str:
    """Render a single ContextBlock as an XML string."""
    section = block.section or ""
    section_id = block.section_id or ""
    entities_str = ", ".join(block.entities) if block.entities else ""
    page_end = str(block.page_end) if block.page_end is not None else ""
    return CONTEXT_BLOCK_XML.format(
        index=block.index,
        chunk_id=_xml_escape(block.chunk_id),
        document=_xml_escape(block.document),
        document_id=_xml_escape(block.document_id),
        section=_xml_escape(section),
        section_id=_xml_escape(section_id),
        module=_xml_escape(block.module),
        page=block.page,
        page_end=page_end,
        entities=_xml_escape(entities_str),
        text=_xml_escape(block.text),
    )


def render_context_xml(blocks: list[ContextBlock]) -> str:
    """Render a list of ContextBlocks as a complete XML context section."""
    parts = ["<context>"]
    for block in blocks:
        parts.append(render_context_block(block))
    parts.append("</context>")
    return "\n".join(parts)


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
