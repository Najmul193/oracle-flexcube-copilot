#!/usr/bin/env bash
set -euo pipefail

PID=$(pgrep -f "streamlit run" 2>/dev/null || true)

if [ -z "$PID" ]; then
    echo "Oracle FLEXCUBE Copilot UI is not running."
    exit 0
fi

echo "Stopping Oracle FLEXCUBE Copilot UI (PID: $PID)..."
kill "$PID" 2>/dev/null || true

# Wait and confirm
sleep 1
if pgrep -f "streamlit run" >/dev/null 2>&1; then
    echo "Force stopping..."
    pkill -9 -f "streamlit run" 2>/dev/null || true
fi

echo "Stopped."
