"""Answer Generator — produces structured answers from PromptRequests."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from oracle_flexcube_copilot.llm.client import OllamaLLMClient
from oracle_flexcube_copilot.llm.models import AnswerMetadata, AnswerResponse, Citation
from oracle_flexcube_copilot.prompting.models import PromptRequest

logger = logging.getLogger("oracle_flexcube_copilot.llm.generator")

_MODE_INSTRUCTIONS = {
    "concise": (
        "\n\nAnswer Mode: Concise\n"
        "Provide a 2-5 sentence answer focusing on the most important information. "
        "Include citations inline (Document, Section, Page)."
    ),
    "detailed": (
        "\n\nAnswer Mode: Detailed\n"
        "Provide a thorough answer with step-by-step explanation. "
        "Use Oracle terminology precisely. "
        "Include all relevant details from the documentation. "
        "Include citations inline (Document, Section, Page)."
    ),
    "expert": (
        "\n\nAnswer Mode: Expert\n"
        "Provide a comprehensive answer including: technical details, "
        "screen names, field descriptions, notes, warnings, and cross-references "
        "from multiple manuals. Compare information across documents if available. "
        "Use exact Oracle terminology. "
        "Include citations inline (Document, Section, Page)."
    ),
}

_CONFIDENCE_THRESHOLDS = [
    ("High", 80.0),
    ("Medium", 50.0),
    ("Low", 0.0),
]


def _calculate_confidence(prompt_request: PromptRequest) -> tuple[str, float]:
    """Calculate answer confidence algorithmically from retrieval signals.

    Factors:
    - Highest retrieval score
    - Average retrieval score
    - Number of supporting chunks
    - Number of unique documents agreeing
    - Entity match presence

    Returns:
        (label, percentage) e.g. ("High", 87.5)
    """
    blocks = prompt_request.context_blocks
    if not blocks:
        return "Low", 0.0

    scores = [b.score for b in blocks]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    unique_docs = len({b.document for b in blocks})
    has_entities = any(bool(b.entities) for b in blocks)

    score_factor = max_score * 0.4 + avg_score * 0.3
    doc_factor = min(unique_docs / 3.0, 1.0) * 0.15
    entity_factor = 0.15 if has_entities else 0.0

    raw = (score_factor + doc_factor + entity_factor) * 100.0
    raw = min(raw, 100.0)

    for label, threshold in _CONFIDENCE_THRESHOLDS:
        if raw >= threshold:
            return label, round(raw, 1)

    return "Low", round(raw, 1)


def _build_system_instruction(mode: str) -> str:
    """Build the mode-specific system instruction appended to prompts."""
    return _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["concise"])


def _build_citations(prompt_request: PromptRequest) -> list[Citation]:
    """Extract unique citations from the prompt request's context blocks."""
    seen: set[tuple[str, str | None, int]] = set()
    citations: list[Citation] = []
    for block in prompt_request.context_blocks:
        key = (block.document, block.section, block.page)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    document=block.document,
                    section=block.section,
                    page=block.page,
                    score=block.score,
                )
            )
    return citations


class RAGAnswerGenerator:
    """Generates answers from PromptRequests using the LLM client.

    Never rebuilds prompts internally — accepts PromptRequest directly.
    Attaches citations from retrieved context (LLM never creates citations).
    Computes confidence algorithmically.
    Supports three answer modes: concise, detailed, expert.
    """

    def __init__(self, client: OllamaLLMClient | None = None) -> None:
        self._client = client or OllamaLLMClient()

    def generate(
        self,
        prompt_request: PromptRequest,
        mode: str = "concise",
    ) -> AnswerResponse:
        """Generate a complete answer from a PromptRequest.

        Args:
            prompt_request: Fully assembled prompt with context and citations.
            mode: Answer mode (concise/detailed/expert).

        Returns:
            AnswerResponse with answer, citations, confidence, and metadata.
        """
        if mode not in _MODE_INSTRUCTIONS:
            logger.warning("Unknown mode '%s', falling back to 'concise'", mode)
            mode = "concise"

        citations = _build_citations(prompt_request)
        confidence_label, confidence_pct = _calculate_confidence(prompt_request)

        full_prompt = self._assemble_prompt(prompt_request, mode)

        t0 = time.perf_counter()
        answer = self._client.generate(full_prompt)
        t1 = time.perf_counter()

        generation_time = t1 - t0

        metadata = AnswerMetadata(
            prompt_tokens=prompt_request.estimated_tokens,
            completion_tokens=self._estimate_tokens(answer),
            total_tokens=prompt_request.estimated_tokens + self._estimate_tokens(answer),
            retrieval_time=0.0,
            generation_time=generation_time,
            model_name=self._client.model_name,
        )

        return AnswerResponse(
            answer=answer,
            citations=citations,
            confidence=confidence_label,
            confidence_percentage=confidence_pct,
            reasoning_time=generation_time,
            metadata=metadata,
            mode=mode,
        )

    def stream(
        self,
        prompt_request: PromptRequest,
        mode: str = "concise",
    ) -> Iterator[str]:
        """Stream tokens from the LLM.

        Args:
            prompt_request: Fully assembled prompt with context and citations.
            mode: Answer mode (concise/detailed/expert).

        Yields:
            Each token as a string.
        """
        if mode not in _MODE_INSTRUCTIONS:
            mode = "concise"

        full_prompt = self._assemble_prompt(prompt_request, mode)
        yield from self._client.stream(full_prompt)

    def _assemble_prompt(self, prompt_request: PromptRequest, mode: str) -> str:
        """Assemble the final prompt by appending mode instruction."""
        mode_instr = _build_system_instruction(mode)
        return f"{prompt_request.formatted_context}\n\n{prompt_request.user_prompt}\n\n{mode_instr}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4
