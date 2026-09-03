#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if available.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Usage: ./scripts/inference_fewshot_gemini.sh [language] [--modes "mode1 mode2"] [--staged]
# Example: ./scripts/inference_fewshot_gemini.sh indo --modes static_diverse --staged
# GEMINI_API_KEY is loaded from .env by the Python runner (dotenv).
MODEL_NAME="${MODEL_NAME:-gemini-3-flash-preview}"
MAX_API_RETRIES="${MAX_API_RETRIES:-5}"
RETRY_BASE_SLEEP_SECONDS="${RETRY_BASE_SLEEP_SECONDS:-2.0}"

language="indo"
STAGED=0
MODES_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged)
            STAGED=1
            shift
            ;;
        --modes)
            MODES_ARG="$2"
            shift 2
            ;;
        --modes=*)
            MODES_ARG="${1#--modes=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            language="$1"
            shift
            ;;
    esac
done

if [[ "$STAGED" == "1" ]]; then
    declare -A MODE_PROMPTSET=(
        [zero]=fewshot_zeroshot_staged
        [static_curated]=exp2_staged
        [static_random]=fewshot_static_random_staged
        [static_diverse]=fewshot_static_diverse_staged
        [dynamic]=fewshot_dynamic_staged
    )
    STAGED_FLAG="--staged"
else
    declare -A MODE_PROMPTSET=(
        [zero]=fewshot_zeroshot
        [static_curated]=exp1
        [static_random]=fewshot_static_random
        [static_diverse]=fewshot_static_diverse
        [dynamic]=fewshot_dynamic
    )
    STAGED_FLAG=""
fi

# Pass --modes "mode1 mode2" to run only a subset of conditions, e.g.
# ./scripts/inference_fewshot_gemini.sh indo --modes static_diverse
read -ra MODES <<< "${MODES_ARG:-zero static_curated static_random static_diverse dynamic}"

for seed in 123 777 2024 9584 31415; do
    for mode in "${MODES[@]}"; do
        python -m src.main.run_agent_fewshot_gemini \
            --model_name "$MODEL_NAME" \
            --test_case_path "dataset/hotel_reviews/$language/mvp_aos/test.json" \
            --prompt_set "${MODE_PROMPTSET[$mode]}" \
            --fewshot_mode "$mode" \
            --num_fewshot 3 \
            --fewshot_seed 42 \
            --embedding_model_name Qwen/Qwen3-Embedding-0.6B \
            --max_api_retries "$MAX_API_RETRIES" \
            --retry_base_sleep_seconds "$RETRY_BASE_SLEEP_SECONDS" \
            --seed $seed \
            $STAGED_FLAG
            # --track_tokens
    done
done
