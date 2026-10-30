#!/bin/bash
# ==============================================================================
# Live Call AI Assistant — macOS Setup & Launch Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "Live Call AI Assistant -- Setup & Launcher"
echo "=========================================================="

APP_SANDBOX="$HOME/.context_assistant"
if [ -d "$SCRIPT_DIR/../venv" ]; then
    VENV_DIR="$SCRIPT_DIR/../venv"
else
    VENV_DIR="$SCRIPT_DIR/.venv"
fi

# 1. Prepare local profile sandboxes
echo "-> Initializing local sandboxes at: $APP_SANDBOX"
mkdir -p "$APP_SANDBOX/profiles/Default_Client"
mkdir -p "$APP_SANDBOX/vector_db"
mkdir -p "$APP_SANDBOX/review_packets"

# 2. Check Python 3
echo "-> Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Error: Python 3 not found. Please install via Homebrew: brew install python"
    exit 1
fi

# 3. Check LLM Engine (Ollama or OpenAI API Key)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "[OK] OPENAI_API_KEY detected in environment (using OpenAI API)."
else
    echo "-> Checking local Ollama engine..."
    if command -v ollama &> /dev/null; then
        if ! curl -s http://localhost:11434/api/tags > /dev/null; then
            echo "[WARN] Ollama server is not active. Attempting to launch..."
            open -a Ollama 2>/dev/null || ollama serve &
            sleep 3
        fi
        echo "[OK] Ollama is running on http://localhost:11434"
    else
        echo "[WARN] Ollama command not detected in PATH and OPENAI_API_KEY is not set."
        echo "   The app will still launch. To enable suggestions, run: brew install ollama && ollama run llama3.1"
    fi
fi

# 4. Check Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "-> Creating isolated virtual environment inside $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 5. Install / Verify Dependencies
if [ "$1" = "--tkinter" ] || [ "$1" = "tkinter" ] || [ "$1" = "--test-tkinter" ] || [ "$1" = "test-tkinter" ] || [ "$1" = "--test" ] || [ "$1" = "test" ]; then
    REQ_FILE="requirements.txt"
else
    REQ_FILE="requirements_aurora.txt"
fi

echo "-> Verifying Python dependencies ($REQ_FILE)..."
pip install -r "$REQ_FILE" --quiet

# 6. Execute Run Command
if [ "$1" = "--test" ] || [ "$1" = "test" ]; then
    echo "=========================================================="
    echo "[RUN] Launching Audio Signal Diagnostics..."
    echo "=========================================================="
    python3 test_audio.py
elif [ "$1" = "--test-e2e" ] || [ "$1" = "test-e2e" ]; then
    echo "=========================================================="
    echo "[RUN] Running Headless E2E Test (Aurora UI + RAG)..."
    echo "=========================================================="
    python3 test_aurora_e2e.py
elif [ "$1" = "--test-tkinter" ] || [ "$1" = "test-tkinter" ]; then
    echo "=========================================================="
    echo "[RUN] Running Headless E2E Test (Elevated Tkinter App)..."
    echo "=========================================================="
    python3 test_tkinter_e2e.py
elif [ "$1" = "--tkinter" ] || [ "$1" = "tkinter" ]; then
    echo "=========================================================="
    echo "[RUN] Launching Elevated Tkinter Live Assistant Window..."
    echo "=========================================================="
    python3 app.py
else
    echo "=========================================================="
    echo "[RUN] Launching Aurora UI Live Assistant Window..."
    echo "=========================================================="
    python3 app_aurora.py
fi
