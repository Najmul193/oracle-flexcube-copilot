# Oracle FLEXCUBE Copilot — Architecture & Technical Reference

## 1. Overview

A Retrieval-Augmented Generation (RAG) system that answers questions about Oracle FLEXCUBE Universal Banking System using 179 official Oracle PDFs as the knowledge base. The system runs entirely locally via Ollama — no cloud dependencies.

- **LLM**: Qwen3:8B via Ollama
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB (chromadb `>=0.5.0`)
- **Sparse Index**: BM25 via `rank-bm25`
- **Entity Index**: SQLite via ChromaDB's metadata
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- **UI**: Streamlit
- **Language**: Python 3.14+, pydantic throughout

---

## 2. Pipeline Architecture

```
                     ┌──────────────────────────────────────┐
                     │         INDEXING PIPELINE            │
                     │                                      │
 PDFs ──►Ingestion──►Enrichment──►Chunking──►Embedding──►Index
  179      parser      headings     semantic   nomic-embed  ChromaDB
           metadata    entities     overlap    text         + BM25
                       tables       800/100                 + Entity
                                                                  │
                                                                  ▼
                     ┌──────────────────────────────────────┐
                     │         RETRIEVAL PIPELINE           │
                     │                                      │
 Query ──►Vector──►RRF──►Prompt Builder──►LLM──►Answer     │
          BM25     Fuse   XML context     Qwen3  +Citations │
          Entity          + system prompt  :8b   +Confidence│
                                                           │
                     └──────────────────────────────────────┘
```

### Pipeline stages

| Stage | Module | Responsibility |
|-------|--------|---------------|
| Ingestion | `ingestion/` | Extract text + metadata from PDFs |
| Enrichment | `enrichment/` | Detect headings, entities, tables, hierarchy |
| Chunking | `chunking/` | Split documents into semantic sections |
| Embedding | `embedding/` | Vectorise chunks via Ollama + LRU cache |
| Indexing | `indexing/` | Store vectors in ChromaDB, build BM25, entity index |
| Retrieval | `retrieval/` | Multi-strategy search + RRF fusion |
| Prompting | `prompting/` | Assemble XML context + system prompt |
| Generation | `llm/` | Call Ollama, stream, format, compute confidence |

---

## 3. Ingestion Pipeline

### 3.1 PDF Parsing (`ingestion/parser.py`)

Three extraction libraries are used in a cascade:

1. **pdfplumber** — Primary extractor. Best table and text extraction.
2. **PyMuPDF** (fitz) — Fallback with structural metadata.
3. **pypdf** — Final fallback.

Each parser reads: text content, page numbers, document metadata (title, author, subject, creation date).

### 3.2 Metadata (`ingestion/metadata.py`)

Document metadata model:

```python
class DocumentMetadata(BaseModel):
    document_id: str          # SHA-256 of content
    document_name: str        # Filename
    title: str | None         # PDF title
    author: str | None        # PDF author
    subject: str | None       # PDF subject
    total_pages: int
    module_classification: str # e.g. "CASA", "General Ledger"
```

### 3.3 Ingestion Service (`ingestion/service.py`)

`DocumentIngestionService` orchestrates:
1. Filesystem scanning (`Scanner`)
2. Document loading (`Loader`)
3. Metadata extraction (`MetadataExtractor`)
4. Deduplication via content hash

Documents are resolved to `Document` objects containing raw text + metadata.

---

## 4. Document Enrichment (`enrichment/`)

Before chunking, each document is enriched with structural information:

| Module | Extracts |
|--------|----------|
| `headings.py` | Section headings with hierarchy level (H1-H6) |
| `tables.py` | Table boundaries and captions |
| `hierarchy.py` | Heading-parent relationships |
| `classification.py` | Oracle module (CASA, GL, TD, etc.) via regex + filename patterns |
| `references.py` | Cross-document references (e.g. "See GL User Guide") |
| `toc.py` | Table of Contents detection |

The enrichment runs before chunking so headings and entities can inform the chunk boundaries.

---

## 5. Semantic Chunking (`chunking/`)

### 5.1 Strategy

`SemanticSectionChunker` in `chunking/strategy.py` splits documents using heading structure:

1. Parse enriched text for heading markers.
2. Split at heading boundaries into logical sections.
3. Apply token budget (`chunk_size=800`, `chunk_overlap=100`).
4. Sections exceeding `chunk_size` are further split with overlap.

### 5.2 Chunk Model

```python
class Chunk(BaseModel):
    id: str                     # UUID
    document_id: str            # Parent document SHA-256
    section_title: str | None
    section_id: str | None
    heading_path: list[str]     # Breadcrumb: ["CASA", "Interest", "Rate Config"]
    page_start: int
    page_end: int | None
    text: str
    oracle_entities: list[EntityRef]
    metadata: ChunkMetadata
```

---

## 6. Embedding (`embedding/`)

### 6.1 Engine (`embedding/engine.py`)

`EmbeddingEngine` wraps Ollama's `nomic-embed-text` model:

- Single text embedding: `embed(text) -> list[float]`
- Batch chunk embedding: `embed_chunks(chunks) -> list[EmbeddedChunk]`
- Query embedding: same model, same dimension (768 for nomic-embed-text)

### 6.2 Cache (`embedding/cache.py`)

`EmbeddingCache` is a disk-backed LRU cache keyed by content hash (SHA-256):

- Cache directory: `cache/embeddings/` (JSON files)
- Automatic hit/miss logic
- Configurable max age
- Thread-safe via file locking

Cache keys are JSON objects containing `{text_hash, model_name, model_version}` to handle model changes.

### 6.3 Embedded Chunk

```python
class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]
    model_name: str
    model_version: str
```

---

## 7. Indexing Layer (`indexing/`)

### 7.1 ChromaDB Vector Index (`indexing/indexer.py`)

`ChromaIndexer` manages the ChromaDB collection:

- Collection name: `oracle-flexcube-v1` (configurable)
- Embedding dimension: 768
- Metadata stored per chunk: document name, page, section, entities, module
- Search returns `SearchResult` objects with cosine distance scores

```python
class ChromaIndexer:
    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]: ...
    def add_chunks(self, chunks: list[EmbeddedChunk]) -> IndexMetrics: ...
    def health_check(self) -> IndexHealth: ...
```

### 7.2 BM25 Sparse Index (`indexing/bm25_indexer.py` / `retrieval/bm25.py`)

- Built during ingestion after all chunks are ready
- Tokenizer: `\w+` regex, lowercased
- Stored as pickle in `data/bm25_index.pkl` (~372 MB for 179 docs)
- Each chunk includes full metadata for `SearchResult` reconstruction

### 7.3 Entity Index (`indexing/entity_index.py`)

SQLite database (`entity_db/`) of Oracle-specific entities:

- Screen codes (e.g. `STDCASA`, `GLSACC`)
- Module names (e.g. `General Ledger`, `CASA`)
- Business terms (e.g. `interest rate`, `maturity date`)
- Cross-references between entities

`EntityRetriever` (`retrieval/entity.py`) queries this index separately and feeds results into the RRF merge.

---

## 8. Retrieval Layer (`retrieval/`)

### 8.1 Structure

| Retriever | Type | Algorithm |
|-----------|------|-----------|
| `VectorRetriever` | Dense | Cosine similarity in ChromaDB |
| `BM25Retriever` | Sparse | BM25Okapi |
| `EntityRetriever` | Structured | SQLite regex + entity matching |

### 8.2 Reciprocal Rank Fusion (`retrieval/fusion.py`)

`RRFFuser` combines multiple result lists using the standard formula:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

Where `k = 60` (standard value) and `rank_i(d)` is document d's rank in list i.

**Properties:**
- Scores from different methods are never compared directly (they use different scales).
- RRF scores are normalised to `[0, 1]` after fusion.
- The `retrieval_method` field tracks which methods found each result (e.g. `"vector+bm25"`).

### 8.3 Fusion Pipeline

```
Vector Retriever  ──►  [rank 1..5]  ──┐
BM25 Retriever    ──►  [rank 1..5]  ──┼──► RRF (k=60) ──► top_k results
Entity Retriever  ──►  [rank 1..5]  ──┘
```

### 8.4 Search Result Model

```python
class SearchResult(BaseModel):
    chunk_id: str
    score: float              # Normalised RRF score
    source_document: str      # Filename
    page: int
    heading: str
    oracle_entities: list[str]
    text: str
    retrieval_method: str     # e.g. "vector", "bm25", "vector+bm25"
    document_id: str
    module: str
    section_id: str
```

---

## 9. Prompting System (`prompting/`)

### 9.1 Architecture

`RAGPromptBuilder` orchestrates assembly of the LLM prompt:

```
SearchResults ──► ContextFormatter
                    │
                    ▼
              XML context string
              + ContextBlock list
                    │
                    ▼
SystemPromptBuilder ──► System prompt text
                            │
                            ▼
                    TokenEstimator
                    (4 chars/token heuristic)
                            │
                            ▼
                    PromptRequest
                    ├── system_prompt
                    ├── user_prompt (original question)
                    ├── formatted_context (XML)
                    ├── context_blocks (structured)
                    ├── estimated_tokens
                    └── citations
```

### 9.2 System Prompt

Two modes (via `SystemPromptBuilder`):

**strict** (default):
```
You are an Oracle FLEXCUBE Copilot — a specialised assistant for Oracle
FLEXCUBE documentation.

Answering Policy
1. Use only the supplied Oracle documentation.
2. If the answer is incomplete, explicitly state that the documentation
   provided is insufficient.
3. Never invent menu names, screen names, parameters, or procedures.
4. Quote Oracle terminology exactly when possible.
5. Always include citations in the format: (Document, Section, Page)
6. If multiple documents disagree, mention the discrepancy instead of
   choosing one.
7. If the user's question is ambiguous, explain the ambiguity and answer
   based on the available context.
```

**benchmark** (minimal for evaluation runs):
```
You are an Oracle FLEXCUBE Copilot — a specialised assistant for Oracle
FLEXCUBE documentation.
```

### 9.3 Context Format

Retrieved chunks are rendered as XML in the prompt:

```xml
<context>
<context_block index="1" id="abc123">
<chunk_id>abc123</chunk_id>
<document>CASA_User_Guide.pdf</document>
<section>Interest Rate Configuration</section>
<page>45</page>
<entities>STDCASA, Interest, Rate</entities>
<text>
...chunk text content...
</text>
</context_block>
<!-- more blocks -->
</context>

Today's Date: 2026-07-01

You are an Oracle FLEXCUBE Copilot...

<context>...</context>

User Question
--------------
How do I configure CASA interest rates?

Answer Mode: Concise
Provide a 2-5 sentence answer focusing on the most important information.
Include citations inline (Document, Section, Page).
```

### 9.4 Token Budget

- Default `max_tokens`: 4096
- `ContextFormatter` adds chunks greedily until the budget is exceeded
- Chunks below `min_score` (default 0.0) are excluded
- Token estimation: `len(text) // 4` (rough heuristic)

---

## 10. LLM Integration (`llm/`)

### 10.1 Ollama Client (`llm/client.py`)

`OllamaLLMClient` wraps the `ollama` Python SDK:

- Constructor verifies model availability
- Automatic retry (3 attempts, exponential backoff: 1s, 4s, 10s)
- Error classification: ConnectionError, TimeoutError, ModelNotFoundError, ContextOverflowError
- Streaming uses Ollama's `stream=True` parameter

```python
class OllamaLLMClient:
    def generate(self, prompt: str, **kwargs) -> str: ...
    def stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
```

### 10.2 RAG Answer Generator (`llm/generator.py`)

`RAGAnswerGenerator`:

1. Takes a `PromptRequest` (never rebuilds prompts internally).
2. Attaches citations from retrieved context (LLM never generates citations).
3. Computes confidence algorithmically from retrieval signals.
4. Appends mode-specific instructions to the prompt.
5. Calls `OllamaLLMClient.generate()` or `.stream()`.

### 10.3 Three Answer Modes

| Mode | Instruction appended to prompt |
|------|-------------------------------|
| `concise` | 2-5 sentence answer, key facts, citations inline |
| `detailed` | Step-by-step with full documentation details |
| `expert` | Technical deep-dive with cross-references, screen names, field descriptions |

### 10.4 Confidence Scoring

Confidence is computed from retrieval signals alone — never from the LLM output:

```
score_factor  = max_score * 0.4 + avg_score * 0.3
doc_factor    = min(unique_docs / 3, 1.0) * 0.15
entity_factor = 0.15 if any chunk has entities else 0.0
raw           = (score_factor + doc_factor + entity_factor) * 100.0

→ ≥80% → High
→ ≥50% → Medium
→ <50% → Low
```

### 10.5 Answer Response

```python
class AnswerResponse(BaseModel):
    answer: str                 # Generated text
    citations: list[Citation]   # From retrieved chunks
    confidence: str             # High / Medium / Low
    confidence_percentage: float  # 0.0–100.0
    reasoning_time: float       # Total seconds
    metadata: AnswerMetadata    # Tokens + timing
    mode: str                   # concise / detailed / expert
```

### 10.6 Streaming

`StreamHandler` wraps the token iterator:

```python
handler = StreamHandler()
for token in handler.handle(generator.stream(prompt_request)):
    print(token, end="")
# handler.text == complete answer
# handler.token_count == total tokens
```

---

## 11. User Interface (`ui/`)

### 11.1 Streamlit App

`ui/app.py` provides a chat interface with:

- `@st.cache_resource` for the RAG pipeline (initialised once per session)
- `st.session_state.messages` for chat history
- Streaming via `st.empty()` + `.markdown()` updates with blinking cursor
- `st.status()` for progress indicator (searching → generating → done)
- Expandable "Sources & Metrics" panel per answer

### 11.2 Efficiency Optimisation

When streaming, the UI avoids a second LLM call for metadata:

1. Stream tokens directly to screen
2. Compute `AnswerResponse` from `PromptRequest` metadata (citations, confidence)
3. Result: one LLM call instead of two (the CLI previously called `generate()` again after streaming)

### 11.3 Sidebar Controls

| Control | Range | Default |
|---------|-------|---------|
| Mode | concise / detailed / expert | concise |
| Top-K | 1–15 | 5 |
| Min Score | 0.0–1.0 | 0.0 |
| Streaming | toggle | on |

---

## 12. Evaluation (`evaluation/`)

### 12.1 Dataset Format

YAML-based Q&A pairs with relevance judgements:

```yaml
queries:
  - question: "How do I configure CASA interest rates?"
    relevant_docs: ["CASA.pdf", "Interest.pdf"]
    module: "CASA"
```

### 12.2 Metrics

| Metric | Description |
|--------|-------------|
| Hit@k | Was any relevant doc in top-k? |
| Recall@k | Proportion of relevant docs retrieved |
| MRR | Mean Reciprocal Rank of first relevant doc |
| NDCG@k | Normalised Discounted Cumulative Gain |

### 12.3 Benchmark Runner

```bash
oracle-copilot benchmark benchmark_dataset.yaml --top-k 10
```

Outputs a markdown report with per-query and aggregate metrics.

---

## 13. Configuration

All configuration via `Settings` (pydantic-settings) loaded from environment or `.env`:

```python
class Settings(BaseSettings):
    project_root: Path = Path.cwd()
    data_dir: Path = Path("data")
    chroma_db_dir: Path = Path("chroma_db")
    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_ctx: int = 8192
    llm_max_tokens: int = 2048
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 5
    retrieval_alpha: float = 0.5
    prompt_max_tokens: int = 4096
    chroma_collection_name: str = "oracle-flexcube-v1"
```

Paths are resolved relative to `project_root`. All path settings have `resolved_*` properties.

---

## 14. Error Handling

### 14.1 LLM Errors (`llm/exceptions.py`)

| Exception | Condition |
|-----------|-----------|
| `LLMConnectionError` | Ollama unreachable / connection refused |
| `LLMTimeoutError` | Request exceeds timeout |
| `LLMModelNotFoundError` | Model not in Ollama's list |
| `LLMContextOverflowError` | Context window exceeded |
| `LLMEmptyResponseError` | Empty response from server |

### 14.2 Retry Strategy

Both `generate()` and `stream()` implement:
- 3 retries
- Exponential backoff: 1s → 4s → 10s
- Only retries connection/timeout errors
- Other errors propagate immediately

### 14.3 Shared Exceptions (`exceptions.py`)

- `CorruptedCacheError` — embedding cache integrity failure
- `ConfigurationError` — invalid settings

---

## 15. Data Flow Summary

```
User Question
       │
       ▼
  ┌─────────────────┐
  │  Embed Query     │  nomic-embed-text → 768-dim vector
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Vector Search   │  ChromaDB cosine similarity
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  BM25 Search     │  Keyword matching
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Entity Search   │  SQLite entity matching
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  RRF Fusion      │  k=60, normalise to [0,1]
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Prompt Builder  │  XML context + system prompt + mode instruction
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Qwen3:8B        │  generate or stream
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  AnswerResponse  │  answer + citations + confidence + metadata
  └─────────────────┘
```

## 16. Directory Layout

```
src/oracle_flexcube_copilot/
├── __init__.py            # Package version
├── cli.py                 # Click CLI (ask, ingest, search, prompt, benchmark, stats)
├── config.py              # Pydantic Settings
├── exceptions.py          # Shared exception classes
├── logger.py              # Logging configuration
├── chunking/
│   ├── interfaces.py      # Chunker protocol
│   ├── models.py          # Chunk, ChunkMetadata
│   └── strategy.py        # SemanticSectionChunker
├── embedding/
│   ├── cache.py           # LRU disk cache
│   ├── engine.py          # EmbeddingEngine (nomic-embed-text)
│   └── models.py          # EmbeddedChunk
├── enrichment/
│   ├── classification.py  # Module classification
│   ├── headings.py        # Heading detection
│   ├── hierarchy.py       # Document hierarchy
│   ├── interfaces.py      # Enricher protocol
│   ├── models.py          # Document enrichment models
│   ├── references.py      # Cross-reference detection
│   ├── service.py         # DocumentEnrichmentService
│   ├── tables.py          # Table detection
│   └── toc.py             # TOC extraction
├── evaluation/
│   ├── benchmark.py       # Benchmark runner
│   ├── dataset.py         # Dataset loading
│   ├── metrics.py         # Hit@k, Recall, MRR, NDCG
│   ├── models.py          # Evaluation models
│   └── report.py          # Markdown report generation
├── indexing/
│   ├── entity_index.py    # SQLite entity index
│   ├── indexer.py         # ChromaIndexer (ChromaDB)
│   └── models.py          # IndexMetrics, IndexHealth, SearchResult
├── ingestion/
│   ├── loader.py          # Document loader
│   ├── metadata.py        # Metadata extraction
│   ├── models.py          # Document, DocumentMetadata
│   ├── parser.py          # PDF text extraction
│   ├── scanner.py         # Filesystem scanner
│   └── service.py         # DocumentIngestionService
├── llm/
│   ├── client.py          # OllamaLLMClient (retry, streaming)
│   ├── engine.py          # (reserved)
│   ├── exceptions.py      # LLM error hierarchy
│   ├── formatter.py       # ConsoleAnswerFormatter
│   ├── generator.py       # RAGAnswerGenerator
│   ├── interfaces.py      # LLMClient, AnswerGenerator, AnswerFormatter protocols
│   ├── models.py          # LLMConfig, Citation, AnswerMetadata, AnswerResponse
│   └── stream.py          # StreamHandler
├── prompting/
│   ├── builder.py         # RAGPromptBuilder
│   ├── formatter.py       # DefaultContextFormatter, HeuristicTokenEstimator
│   ├── interfaces.py      # ContextFormatter protocol
│   ├── models.py          # ContextBlock, ContextConfig, PromptRequest
│   └── templates.py       # SystemPromptBuilder, XML rendering
├── prompts/
│   ├── __init__.py
│   └── builder.py         # (legacy alternative prompt builder)
├── retrieval/
│   ├── bm25.py            # BM25Indexer, BM25Retriever
│   ├── engine.py          # VectorRetriever
│   ├── entity.py          # EntityRetriever
│   └── fusion.py          # RRFFuser
└── ui/
    ├── __init__.py
    └── app.py             # Streamlit interface

tests/                     # 45 test files mirroring src structure
docs/                      # 179 Oracle PDFs (~1.2 GB)
data/                      # bm25_index.pkl (~372 MB, gitignored)
chroma_db/                 # ChromaDB persistent storage (gitignored)
cache/                     # Embedding cache (gitignored)
entity_db/                 # SQLite entity index (gitignored)
```

---

## 17. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid retrieval** (dense + sparse) | Dense captures semantics; sparse captures exact Oracle terminology that embeddings may miss |
| **RRF over weighted sum** | RRF is parameter-free and handles score distribution mismatches between methods |
| **XML context format** | Structured context makes it easy for the LLM to distinguish document/section/page from text |
| **Confidence from retrieval only** | Avoids cost of a second LLM call; retrieval signals correlate well with answer quality |
| **Three answer modes** | Different users need different depth: quick lookup → detailed steps → expert analysis |
| **No LLM-generated citations** | Prevents hallucinated citations; citations come exclusively from retrieved chunks |
| **4-char token heuristic** | Simple, fast, good enough for budget estimation without a tokeniser |
| **Streamlit for UI** | Rapid development, built-in chat components, hot-reloading |
| **Content-hash caching** | Embedding cache survives re-runs; avoids recomputing unchanged chunk embeddings |
| **pydantic throughout** | Runtime type safety, serialisation, and configuration validation |
