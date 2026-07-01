# oracle-flexcube-copilot
# Oracle FLEXCUBE Copilot

A RAG (Retrieval-Augmented Generation) copilot for Oracle FLEXCUBE documentation, powered by Qwen3:8B via Ollama and ChromaDB.

## Architecture

```
Oracle PDFs (179)
       │
       ▼
  ChromaDB (vector) ◄──► BM25 (keyword) ◄──► Entity Index (regex)
       │                      │                       │
       └──────────────────────┴───────────────────────┘
                              │
                              ▼
                     RRF Fusion (reciprocal rank)
                              │
                              ▼
                       Prompt Builder
                              │
                              ▼
                      Qwen3:8B (Ollama)
                              │
                              ▼
                     Answer + Citations
```

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) with `qwen3:8b` and `nomic-embed-text` models
- `uv` (recommended) or `pip`

## Setup

```bash
# Clone and enter the project
cd oracle-flexcube-copilot

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Or with pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Ingest PDFs into the vector store
oracle-copilot ingest --data-dir data/

# Ask a question (default: concise mode, streaming)
oracle-copilot ask "How do I configure CASA interest rates?"
```

## CLI Commands

### `ask` — Ask questions about Oracle FLEXCUBE

```bash
oracle-copilot ask "How do I maintain GL Balance Transfer?"

# Answer modes
oracle-copilot ask "question" --mode concise      # 2-5 sentences (default)
oracle-copilot ask "question" --mode detailed      # Full step-by-step
oracle-copilot ask "question" --mode expert        # Technical + cross-references

# Streaming
oracle-copilot ask "question" --stream             # Real-time token streaming (default)
oracle-copilot ask "question" --no-stream          # Wait for full answer

# Minimum confidence threshold
oracle-copilot ask "question" --min-score 0.6      # Only answer if confidence >= 60%
```

**Answer modes:**
| Mode | Description |
|------|-------------|
| `concise` | 2-5 sentence summary with key points and screen codes |
| `detailed` | Step-by-step instructions with all navigation details |
| `expert` | Technical deep-dive with cross-references, accounting entries, and batch processes |

### `ingest` — Index PDF documents

```bash
oracle-copilot ingest --data-dir data/    # Index all PDFs in directory
oracle-copilot ingest --file doc.pdf      # Index a single PDF
```

### `benchmark` — Evaluate retrieval quality

```bash
oracle-copilot benchmark benchmark_dataset.yaml --top-k 10
```

Evaluates Hit@k, Recall@k, MRR, and NDCG@k against a YAML dataset of Q&A pairs.

## LLM Configuration

Set via environment variables or `config.toml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Model name |
| `LLM_TEMPERATURE` | `0.1` | Response creativity |
| `LLM_TOP_P` | `0.9` | Nucleus sampling |
| `LLM_MAX_TOKENS` | `1024` | Max response tokens |
| `LLM_CONTEXT_LENGTH` | `8192` | Context window size |
| `LLM_TIMEOUT` | `120` | Request timeout (seconds) |

## Development

```bash
# Lint
make lint

# Type check
make typecheck

# Run tests
make test
```

## Project Structure

```
oracle-flexcube-copilot/
├── src/oracle_flexcube_copilot/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (ask, ingest, benchmark)
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging setup
│   ├── ingestion/           # PDF ingestion pipeline
│   │   ├── parser.py        # PDF text extraction
│   │   ├── chunker.py       # Semantic chunking with headings
│   │   └── service.py       # Orchestration
│   ├── indexing/            # Vector store & BM25
│   │   ├── chroma.py        # ChromaDB integration
│   │   └── bm25.py          # BM25 keyword index
│   ├── retrieval/           # Hybrid retrieval
│   │   ├── fusion.py        # RRF fusion
│   │   └── entity.py        # Oracle entity extraction
│   ├── prompting/           # Prompt building
│   │   └── builder.py       # RAG prompt construction
│   ├── evaluation/          # Benchmark pipeline
│   │   ├── benchmark.py     # Batch evaluator
│   │   └── metrics.py       # Hit@k, NDCG@k, MRR
│   └── llm/                 # LLM interaction layer
│       ├── client.py        # Ollama API client (retry, streaming)
│       ├── generator.py     # RAG answer generation (3 modes)
│       ├── formatter.py     # Console output formatting
│       ├── stream.py        # Token stream tracking
│       ├── models.py        # Config & response models
│       └── exceptions.py    # Error classification
├── data/                    # PDF files
├── chroma_db/               # Vector store persistence
├── cache/                   # Cached embeddings
├── logs/                    # Application logs
├── docs/                    # Documentation
└── tests/                   # Test suite