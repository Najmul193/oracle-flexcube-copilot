# Oracle FLEXCUBE Copilot

RAG-powered assistant for Oracle FLEXCUBE documentation. Uses hybrid retrieval (vector + BM25 + entity + RRF) over 179 Oracle PDFs to answer questions via Qwen3:8B (Ollama).

## Architecture

```
Oracle PDFs (179)
       │
       ▼
  ┌──────────────────────────────────────────┐
  │  Ingestion Pipeline                      │
  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
  │  │ Extract │→│ Enrich   │→│ Chunk     │ │
  │  │ (pdfpl  │ │ (headings│ │ (semantic │ │
  │  │  umber) │ │ entities)│ │  sections)│ │
  │  └─────────┘ └──────────┘ └───────────┘ │
  └──────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  Index Layer                                │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
  │  │ ChromaDB │  │   BM25   │  │  Entity   │ │
  │  │ (dense)  │  │ (sparse) │  │  (SQLite) │ │
  │  └──────────┘  └──────────┘  └───────────┘ │
  └─────────────────────────────────────────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
           RRF Fusion (k=60)
                      │
                      ▼
             ┌────────────────┐
             │  Prompt Builder │
             │  (XML context)  │
             └────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │  Qwen3:8B      │
             │  (Ollama)      │
             └────────────────┘
                      │
                      ▼
             Answer + Citations + Confidence
```

## Prerequisites

- **Python 3.14+**
- **Ollama** with `qwen3:8b` and `nomic-embed-text`
- ~8 GB RAM (for LLM + vector store)

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Ingest PDFs (from docs/)
oracle-copilot ingest docs/

# Ask a question (CLI)
oracle-copilot ask "How do I configure CASA interest rates?"

# Or launch the UI
oracle-copilot-ui     # --or--
make ui               # --or--
streamlit run src/oracle_flexcube_copilot/ui/app.py
```

## CLI Reference

### `ask` — Question answering

```bash
oracle-copilot ask "How do I maintain GL Balance Transfer?"

# Answer modes
oracle-copilot ask "question" --mode concise      # 2-5 sentences (default)
oracle-copilot ask "question" --mode detailed      # Full step-by-step
oracle-copilot ask "question" --mode expert        # Technical + cross-refs

# Streaming
oracle-copilot ask "question" --stream             # Token streaming (default)
oracle-copilot ask "question" --no-stream          # Wait for full answer

# Confidence filter
oracle-copilot ask "question" --min-score 0.5      # Minimum relevance threshold
```

| Mode | Description |
|------|-------------|
| `concise` | 2-5 sentence summary with key points |
| `detailed` | Step-by-step instructions with navigation |
| `expert` | Technical deep-dive with cross-references |

### `ingest` — Index documents

```bash
oracle-copilot ingest path/to/pdf/directory/
```

Runs: PDF extraction → enrichment (headings, entities, tables) → semantic chunking → embedding → ChromaDB + BM25 index build.

### `prompt` — Inspect the assembled prompt

```bash
oracle-copilot prompt "How do I configure CASA?" --show-context --show-system
```

Builds and displays the full prompt sent to the LLM — useful for debugging (no LLM call).

### `search` — Pure retrieval

```bash
oracle-copilot search "interest rate configuration" --top-k 10
```

Returns raw fused results without LLM generation.

### `benchmark` — Evaluation

```bash
oracle-copilot benchmark benchmark_dataset.yaml --top-k 10
```

Measures Hit@k, Recall@k, MRR, NDCG@k.

### `stats` — System health

```bash
oracle-copilot stats
```

## User Interface

A Streamlit web UI provides the same features with a chat interface:

```bash
make ui
# opens at http://localhost:8501
```

**Features:**
- Chat interface with message history
- Streaming token display
- Mode selector (concise / detailed / expert)
- Top-K slider, Min Score filter
- Per-answer Sources & Metrics expander (citations, confidence, timing, tokens)
- "Clear Chat" button

## Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_MODEL` | `qwen3:8b` | Generation model |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_TOP_P` | `0.9` | Nucleus sampling |
| `LLM_REPEAT_PENALTY` | `1.1` | Token repeat penalty |
| `LLM_NUM_CTX` | `8192` | Context window size |
| `LLM_MAX_TOKENS` | `2048` | Max response tokens |
| `LLM_TIMEOUT` | `120` | Request timeout (s) |
| `CHUNK_SIZE` | `800` | Chunk token target |
| `CHUNK_OVERLAP` | `100` | Chunk overlap tokens |
| `TOP_K_RETRIEVAL` | `5` | Default top-K |
| `RETRIEVAL_ALPHA` | `0.5` | Dense/sparse balance |
| `PROMPT_MAX_TOKENS` | `4096` | Max prompt budget |
| `PROMPT_MIN_SCORE` | `0.0` | Min retrieval score |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `text` | `text` or `json` |

## Development

```bash
make install       # Install dependencies
make lint          # Ruff check + format
make typecheck     # mypy
make test          # pytest + coverage
make test-cov      # pytest with coverage report
make clean         # Remove artifacts
```

### Project structure

```
src/oracle_flexcube_copilot/
├── __init__.py
├── cli.py                 # CLI entry point (click)
├── config.py              # Pydantic settings
├── exceptions.py          # Shared exceptions
├── logger.py              # Logging config
├── chunking/              # Semantic section chunking
├── embedding/             # nomic-embed-text via Ollama + caching
├── enrichment/            # Headings, entities, tables, hierarchy extraction
├── evaluation/            # Benchmark runner, metrics, reporting
├── indexing/              # ChromaDB vector store, BM25, entity index
├── ingestion/             # PDF parsing, metadata, loading
├── llm/                   # Ollama client, RAG generator, formatter, streaming
├── prompting/             # Prompt builder, XML context, system templates
├── prompts/               # Alternative prompt strategies (legacy)
├── retrieval/             # Vector, BM25, entity retrievers, RRF fusion
└── ui/                    # Streamlit chat interface
```

### Adding evaluation

```bash
oracle-copilot benchmark my_dataset.yaml --top-k 15
```

Dataset format (YAML):

```yaml
queries:
  - question: "How do I configure CASA?"
    relevant_docs: ["CASA.pdf", "Interest.pdf"]
    module: "CASA"
```

## License

MIT
