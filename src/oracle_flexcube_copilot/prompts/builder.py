"""Structured Prompt Builder."""

from oracle_flexcube_copilot.indexing.models import SearchResult


class PromptBuilder:
    """Builds strict, highly structured prompts for Oracle documentation QA."""

    SYSTEM_PROMPT = (
        "You are an expert Oracle FLEXCUBE documentation assistant.\n"
        "Your absolute priority is accuracy. You must adhere perfectly to these rules:\n"
        "1. Answer ONLY using the information provided in the 'Retrieved Context' below.\n"
        "2. If the context does not contain the answer, explicitly state: 'The provided documentation does not contain the answer.' Do not invent or guess Oracle functionality.\n"
        "3. Always cite the document name, section, and page number for every claim you make.\n"
        "4. Be concise, direct, and professional.\n"
    )

    def build_rag_prompt(self, question: str, context: list[SearchResult]) -> str:
        """Construct the full RAG prompt using the retrieved context."""
        
        prompt_parts = [self.SYSTEM_PROMPT]
        
        prompt_parts.append("\n----------------\nRetrieved Context\n----------------")
        
        if not context:
            prompt_parts.append("No context retrieved.")
        else:
            for i, result in enumerate(context, 1):
                prompt_parts.append(f"\n[Source {i}]")
                prompt_parts.append(f"Document: {result.source_document}")
                prompt_parts.append(f"Page: {result.page}")
                prompt_parts.append(f"Section: {result.heading}")
                prompt_parts.append(f"Text:\n{result.text.strip()}")
                
        prompt_parts.append("\n----------------\nUser Question\n----------------")
        prompt_parts.append(question.strip())
        
        return "\n".join(prompt_parts)
