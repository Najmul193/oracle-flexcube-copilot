#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check prerequisites
if ! command -v ollama &>/dev/null; then
    echo "Error: Ollama is not installed. Install it from https://ollama.ai" >&2
    exit 1
fi

# Check Ollama is running
if ! ollama list &>/dev/null; then
    echo "Error: Ollama is not running. Start it with: ollama serve" >&2
    exit 1
fi

# Check required models
for model in "qwen3:8b" "nomic-embed-text"; do
    if ! ollama list 2>/dev/null | grep -q "$model"; then
        echo "Pulling model: $model..."
        ollama pull "$model"
    fi
done

# Activate virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install -e ".[dev]" --quiet
fi

# Launch UI
echo "Starting Oracle FLEXCUBE Copilot UI..."
echo "Open http://localhost:8501 in your browser."
"$VENV_DIR/bin/streamlit" run src/oracle_flexcube_copilot/ui/app.py
