# oracle-flexcube-copilot
# Oracle FLEXCUBE Copilot

A RAG (Retrieval-Augmented Generation) copilot for Oracle FLEXCUBE documentation, powered by Qwen3:8B via Ollama and ChromaDB.

## Architecture

```
Oracle PDFs (179)
       │
       ▼
  ChromaDB + BM25 Index
       │
       ▼
  Relevant Chunks
       │
       ▼
  Prompt Builder
       │
       ▼
  Qwen3:8B (Ollama)
       │
       ▼
     Answer
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

# Ask a question
oracle-copilot ask "How do I configure CASA interest rates?"
```

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
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   ├── exceptions.py      # Custom exceptions
│   ├── ingestion/         # PDF ingestion pipeline
│   ├── retrieval/         # Hybrid retrieval (ChromaDB + BM25)
│   ├── prompt/            # Prompt building
│   └── llm/               # LLM interaction
├── data/                  # PDF files
├── chroma_db/             # Vector store persistence
├── cache/                 # Cached embeddings
├── logs/                  # Application logs
├── docs/                  # Documentation
└── tests/                 # Test suite