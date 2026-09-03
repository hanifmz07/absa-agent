#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: ./scripts/inference_fewshot_loop_gemini.sh [--modes \"mode1 mode2\"] [--staged] <language1> [language2 ...]"
    echo "Example: ./scripts/inference_fewshot_loop_gemini.sh --modes static_diverse indo sun min"
    exit 1
fi

# Gemini is a hosted API, not GPU-bound, so (unlike inference_fewshot_loop.sh)
# every language runs concurrently rather than being batched across a fixed
# GPU count. Override the per-language rate-limit knobs via MODEL_NAME /
# MAX_API_RETRIES / RETRY_BASE_SLEEP_SECONDS env vars if you hit 429s running
# many languages at once, e.g.
# MAX_API_RETRIES=8 ./scripts/inference_fewshot_loop_gemini.sh indo sun min

# Forwarded to each inference_fewshot_gemini.sh invocation, e.g.
# ./scripts/inference_fewshot_loop_gemini.sh --modes static_diverse --staged indo sun min
FEWSHOT_ARGS=()
languages=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged)
            FEWSHOT_ARGS+=(--staged)
            shift
            ;;
        --modes)
            FEWSHOT_ARGS+=(--modes "$2")
            shift 2
            ;;
        --modes=*)
            FEWSHOT_ARGS+=(--modes "${1#--modes=}")
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            languages+=("$1")
            shift
            ;;
    esac
done

if [[ ${#languages[@]} -eq 0 ]]; then
    echo "Error: at least one language is required" >&2
    exit 1
fi

mkdir -p logs

pids=()

for language in "${languages[@]}"; do
    log_file="logs/fewshot_gemini_${language}.log"
    echo "Running Gemini few-shot inference for language: $language (log: $log_file)"
    ./scripts/inference_fewshot_gemini.sh "$language" "${FEWSHOT_ARGS[@]}" > "$log_file" 2>&1 &
    pids+=($!)
done

overall_status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Finished language: ${languages[$i]}"
    else
        echo "FAILED language: ${languages[$i]} (see logs/fewshot_gemini_${languages[$i]}.log)" >&2
        overall_status=1
    fi
done

echo "----------------------------------------"
if [[ $overall_status -eq 0 ]]; then
    echo "All language few-shot inferences completed."
else
    echo "Some language few-shot inferences FAILED — check logs/ above." >&2
fi
exit "$overall_status"
