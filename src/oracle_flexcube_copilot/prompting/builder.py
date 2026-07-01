from __future__ import annotations

import logging

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.formatter import (
    DefaultContextFormatter,
    HeuristicTokenEstimator,
)
from oracle_flexcube_copilot.prompting.models import ContextConfig, PromptRequest
from oracle_flexcube_copilot.prompting.templates import SystemPromptBuilder, render_date

logger = logging.getLogger("oracle_flexcube_copilot.prompting.builder")


class RAGPromptBuilder:
    """Assembles a complete PromptRequest from a question and retrieved context.

    Orchestrates:
    - SystemPromptBuilder for instructions
    - DefaultContextFormatter for XML context assembly
    - HeuristicTokenEstimator for token accounting
    """

    def __init__(
        self,
        system_builder: SystemPromptBuilder | None = None,
        formatter: DefaultContextFormatter | None = None,
        estimator: HeuristicTokenEstimator | None = None,
    ) -> None:
        self._system_builder = system_builder or SystemPromptBuilder()
        self._formatter = formatter or DefaultContextFormatter()
        self._estimator = estimator or HeuristicTokenEstimator()

    def build(
        self,
        question: str,
        context: list[SearchResult],
        config: ContextConfig | None = None,
    ) -> PromptRequest:
        """Build a fully structured prompt.

        Args:
            question: The user's original question.
            context: Retrieved search results.
            config: Optional context configuration overrides.

        Returns:
            A PromptRequest ready for the LLM.
        """
        cfg = config or ContextConfig()

        # 1. Build system prompt
        system_prompt = self._system_builder.build() if cfg.include_system_prompt else ""

        # 2. Format context (with token budget enforcement and score filtering)
        xml_context, blocks, citations = self._formatter.format_with_budget(
            context,
            max_tokens=cfg.max_tokens,
            min_score=cfg.min_score,
        )

        # 3. Assemble the full prompt text
        prompt_parts: list[str] = []

        if cfg.include_date:
            prompt_parts.append(f"Today's Date: {render_date()}\n")

        if system_prompt:
            prompt_parts.append(system_prompt)

        if xml_context:
            prompt_parts.append(xml_context)

        prompt_parts.append("User Question\n--------------")
        prompt_parts.append(question.strip())

        full_prompt = "\n\n".join(prompt_parts)
        estimated_tokens = self._estimator.estimate(full_prompt)

        return PromptRequest(
            system_prompt=system_prompt,
            user_prompt=question.strip(),
            formatted_context=xml_context,
            context_blocks=blocks,
            estimated_tokens=estimated_tokens,
            citations=citations,
        )
