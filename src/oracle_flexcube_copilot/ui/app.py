"""Streamlit UI — chat interface for Oracle FLEXCUBE Copilot."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import streamlit as st

from oracle_flexcube_copilot import __version__
from oracle_flexcube_copilot.config import settings
from oracle_flexcube_copilot.embedding.cache import EmbeddingCache
from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer
from oracle_flexcube_copilot.llm import RAGAnswerGenerator
from oracle_flexcube_copilot.llm.models import AnswerMetadata, AnswerResponse
from oracle_flexcube_copilot.llm.models import Citation as AnswerCitation
from oracle_flexcube_copilot.llm.stream import StreamHandler
from oracle_flexcube_copilot.prompting.builder import RAGPromptBuilder
from oracle_flexcube_copilot.prompting.models import ContextBlock, ContextConfig
from oracle_flexcube_copilot.retrieval.bm25 import BM25Retriever
from oracle_flexcube_copilot.retrieval.engine import VectorRetriever
from oracle_flexcube_copilot.retrieval.fusion import RRFFuser

# ── Live model monitoring ──────────────────────────────────────────────


def _ollama_ps() -> str:
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        return result.stdout or result.stderr or "(no output)"
    except FileNotFoundError:
        return "Ollama not found"
    except subprocess.TimeoutExpired:
        return "ollama ps timed out"
    except Exception as e:
        return f"Error: {e}"


st.set_page_config(
    page_title="Oracle FLEXCUBE Copilot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Confidence heuristic (mirrors llm.generator._calculate_confidence) ─

_CONFIDENCE_THRESHOLDS: list[tuple[str, float]] = [
    ("High", 80.0),
    ("Medium", 50.0),
    ("Low", 0.0),
]


def _confidence_from_blocks(
    blocks: list[ContextBlock],
) -> tuple[str, float]:
    """Score confidence from retrieved context blocks."""
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

    raw = min((score_factor + doc_factor + entity_factor) * 100.0, 100.0)
    for label, threshold in _CONFIDENCE_THRESHOLDS:
        if raw >= threshold:
            return label, round(raw, 1)
    return "Low", round(raw, 1)


# ── Cached pipeline initialisation ─────────────────────────────────────


@st.cache_resource
def _init_pipeline() -> dict[str, Any]:
    indexer = ChromaIndexer()
    cache = EmbeddingCache(cache_dir=settings.resolved_cache_dir)
    embedder = EmbeddingEngine(cache=cache)
    return {
        "vector_retriever": VectorRetriever(embedder=embedder, indexer=indexer),
        "bm25_retriever": BM25Retriever(),
        "fuser": RRFFuser(),
        "builder": RAGPromptBuilder(),
        "generator": RAGAnswerGenerator(),
    }


# ── Metadata rendering ─────────────────────────────────────────────────


def _render_metadata(response: AnswerResponse) -> None:
    with st.expander("Sources & Metrics", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            if response.citations:
                st.markdown("**Sources**")
                for c in response.citations:
                    section_str = f" · *{c.section}*" if c.section else ""
                    st.markdown(
                        f"- {c.document} (Page {c.page}){section_str} — Score: `{c.score:.4f}`"
                    )

        with col2:
            st.markdown("**Metrics**")
            color = {"High": "green", "Medium": "orange", "Low": "red"}.get(
                response.confidence, "gray"
            )
            st.markdown(
                f"Confidence: **:{color}[{response.confidence}]** "
                f"({response.confidence_percentage:.0f}%)"
            )

            m = response.metadata
            st.markdown(
                f"- Retrieval: `{m.retrieval_time * 1000:.0f}ms`\n"
                f"- Generation: `{m.generation_time * 1000:.0f}ms`\n"
                f"- Total: `{(m.retrieval_time + m.generation_time) * 1000:.0f}ms`\n"
                f"- Tokens: {m.prompt_tokens} ⟶ {m.completion_tokens} "
                f"({m.total_tokens} total)\n"
                f"- Model: `{m.model_name}`"
            )


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📘 FLEXCUBE Copilot")
    st.caption(f"v{__version__}")

    st.divider()

    st.subheader("Answer Settings")
    mode = st.selectbox(
        "Mode",
        options=["concise", "detailed", "expert"],
        index=0,
        help="Concise 2-5 sentences | Detailed full explanation | Expert technical deep-dive",
    )
    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=15,
        value=settings.top_k_retrieval,
        help="Number of context chunks to retrieve",
    )
    min_score = st.slider(
        "Min Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Minimum relevance score threshold",
    )
    use_streaming = st.toggle("Streaming", value=True)

    st.divider()

    st.subheader("Pipeline")
    st.text(f"LLM: {settings.llm_model}")
    st.text(f"Embedding: {settings.embedding_model}")
    st.text(f"Ollama: {settings.ollama_base_url}")
    st.text(f"Chunks: {settings.chunk_size} tok")
    st.text(f"Overlap: {settings.chunk_overlap} tok")

    st.divider()

    # ── Debug Console ──────────────────────────────────────────────

    with st.expander("🛠 Debug Console", expanded=False):
        _ollama_ps_refresh = st.button("⟳ Refresh Ollama PS", use_container_width=True)
        if _ollama_ps_refresh or "_ollama_ps_cache" not in st.session_state:
            st.session_state["_ollama_ps_cache"] = _ollama_ps()
        st.code(st.session_state["_ollama_ps_cache"], language="text")

        st.markdown("**Live Model Activity**")
        dbg = st.session_state.setdefault("_debug", {})
        c1, c2 = st.columns(2)
        dbg["speed_ph"] = c1.empty()
        dbg["count_ph"] = c2.empty()
        dbg["prompt_ph"] = st.empty()
        dbg["retrieval_ph"] = st.empty()

    st.divider()

    if st.button("Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state["_debug"] = {}
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Main chat area ─────────────────────────────────────────────────────

st.title("Oracle FLEXCUBE Copilot")
st.caption("Ask questions about Oracle FLEXCUBE documentation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "answer" in msg:
            _render_metadata(msg["answer"])

# ── Handle new question ────────────────────────────────────────────────

if prompt := st.chat_input("Ask about Oracle FLEXCUBE..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pipeline = _init_pipeline()

        answer_slot = st.empty()
        status_slot = st.status("Searching documentation...")

        try:
            # 1. Retrieve
            t0 = time.perf_counter()
            vector_results = pipeline["vector_retriever"].retrieve(prompt, top_k=top_k)
            bm25_results = pipeline["bm25_retriever"].retrieve(prompt, top_k=top_k)
            results = pipeline["fuser"].fuse([vector_results, bm25_results], top_k=top_k)
            retrieval_time = time.perf_counter() - t0

            # Debug: retrieval details
            dbg = st.session_state.setdefault("_debug", {})
            retrieval_lines = [f"**Top-{len(results)} chunks** (from vector + BM25)"]
            for r in results:
                retrieval_lines.append(
                    f"- {r.source_document} p.{r.page}  `{r.score:.4f}`  [{r.retrieval_method}]"
                )
            dbg["retrieval_ph"].markdown("\n".join(retrieval_lines))

            if not results:
                status_slot.update(state="error", label="No relevant docs found")
                msg = "I couldn't find relevant documentation. Please rephrase your question."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()

            status_slot.update(state="running", label="Generating answer...")

            # 2. Build prompt
            config = ContextConfig(
                max_tokens=settings.prompt_max_tokens,
                min_score=min_score,
            )
            prompt_req = pipeline["builder"].build(prompt, results, config=config)

            # Debug: raw prompt preview
            full_prompt = f"{prompt_req.formatted_context}\n\n{prompt_req.user_prompt}"
            dbg["prompt_ph"].code(
                full_prompt[:2000] + ("..." if len(full_prompt) > 2000 else ""),
                language="xml",
            )

            # 3. Generate
            if use_streaming:
                handler = StreamHandler()
                token_stream = pipeline["generator"].stream(prompt_req, mode=mode)

                text = ""
                gen_start = time.perf_counter()
                for token in handler.handle(token_stream):
                    text += token
                    answer_slot.markdown(text + "▌")
                    # Live token speed
                    elapsed = time.perf_counter() - gen_start
                    speed = handler.token_count / elapsed if elapsed > 0 else 0
                    dbg["speed_ph"].metric("Speed", f"{speed:.1f} tok/s")
                    dbg["count_ph"].metric("Tokens", str(handler.token_count))
                answer_slot.markdown(text)

                # Build AnswerResponse without a second LLM call
                label, pct = _confidence_from_blocks(prompt_req.context_blocks)

                seen: set[tuple[str, str | None, int]] = set()
                citations = []
                for c in prompt_req.citations:
                    key = (c.document, c.section, c.page)
                    if key not in seen:
                        seen.add(key)
                        citations.append(
                            AnswerCitation(
                                document=c.document,
                                section=c.section,
                                page=c.page,
                                score=c.score,
                            )
                        )

                answer = AnswerResponse(
                    answer=handler.text,
                    citations=citations,
                    confidence=label,
                    confidence_percentage=pct,
                    reasoning_time=retrieval_time,
                    metadata=AnswerMetadata(
                        prompt_tokens=prompt_req.estimated_tokens,
                        completion_tokens=handler.token_count,
                        total_tokens=prompt_req.estimated_tokens + handler.token_count,
                        retrieval_time=retrieval_time,
                        generation_time=0.0,
                        model_name=settings.llm_model,
                    ),
                    mode=mode,
                )
            else:
                t1 = time.perf_counter()
                answer = pipeline["generator"].generate(prompt_req, mode=mode)
                elapsed = time.perf_counter() - t1
                answer.metadata.retrieval_time = retrieval_time
                answer.metadata.generation_time = elapsed

                # Debug: non-streaming stats
                speed = answer.metadata.completion_tokens / elapsed if elapsed > 0 else 0
                dbg["speed_ph"].metric("Speed", f"{speed:.1f} tok/s")
                dbg["count_ph"].metric("Tokens", str(answer.metadata.completion_tokens))

                answer_slot.markdown(answer.answer)

            status_slot.update(state="complete", label="Done")

            _render_metadata(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer.answer, "answer": answer}
            )

        except Exception as e:
            status_slot.update(state="error", label=f"Error: {e}")
            st.error(f"An error occurred: {e}", icon="🚨")


def main() -> None:
    """Entry point for `streamlit run`."""
    pass
