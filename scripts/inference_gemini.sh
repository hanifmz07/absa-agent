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
MAX_RETRIES="${MAX_RETRIES:-3}"
SEED="${SEED:-42}"
TRACK_TOKENS="${TRACK_TOKENS:-true}"
MAX_API_RETRIES="${MAX_API_RETRIES:-5}"
RETRY_BASE_SLEEP_SECONDS="${RETRY_BASE_SLEEP_SECONDS:-2.0}"

CMD=(
    python -m src.main.run_agent_async_gemini
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

# GEMINI_API_KEY is loaded from .env by the Python runner (dotenv).
"${CMD[@]}"

# Example sweep (uncomment to run):
# for max_retries in 1 3 5 10; do
#     for seed in 42 123 777 2024 31415; do
#         python -m src.main.run_agent_async_gemini \
#             --model_name gemini-3-flash-preview \
#             --test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
#             --max_retries "$max_retries" \
#             --prompt_set exp1 \
#             --seed "$seed" \
#             --track_tokens
#     done
# done
