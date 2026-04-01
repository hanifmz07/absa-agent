#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Default run configuration (can be overridden via environment variables).
MODEL_NAME="${MODEL_NAME:-gemini-3-flash-preview}"
TEST_CASE_PATH="${TEST_CASE_PATH:-dataset/hotel_reviews/indo/mvp_aos/test.json}"
PROMPT_SET="${PROMPT_SET:-exp1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
SEED="${SEED:-42}"
TRACK_TOKENS="${TRACK_TOKENS:-true}"
MAX_API_RETRIES="${MAX_API_RETRIES:-50000}"
RETRY_BASE_SLEEP_SECONDS="${RETRY_BASE_SLEEP_SECONDS:-10.0}"
LIMIT="${LIMIT:-}"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: neither 'python' nor 'python3' is available in PATH." >&2
    exit 1
fi

CMD=(
    "$PYTHON_BIN" -m src.main.run_agent_sequential_gemini
    --model_name "$MODEL_NAME"
    --test_case_path "$TEST_CASE_PATH"
    --prompt_set "$PROMPT_SET"
    --max_retries "$MAX_RETRIES"
    --seed "$SEED"
    --max_api_retries "$MAX_API_RETRIES"
    --retry_base_sleep_seconds "$RETRY_BASE_SLEEP_SECONDS"
)

if [ "$TRACK_TOKENS" = "true" ]; then
    CMD+=(--track_tokens)
fi

if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

# GEMINI_API_KEY is loaded from .env by the Python runner (dotenv).
"${CMD[@]}"
