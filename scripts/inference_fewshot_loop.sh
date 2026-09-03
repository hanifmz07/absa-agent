#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: ./scripts/inference_fewshot_loop.sh [--modes \"mode1 mode2\"] [--staged] <language1> [language2 ...]"
    echo "Example: ./scripts/inference_fewshot_loop.sh --modes static_diverse indo sun min"
    exit 1
fi

# GPUs to distribute languages across (one language per GPU, run
# concurrently). Override for other hardware, e.g.
# GPU_IDS="0 1 2 3" ./scripts/inference_fewshot_loop.sh indo sun min jav bug
read -ra GPU_IDS <<< "${GPU_IDS:-0 1}"
NUM_GPUS=${#GPU_IDS[@]}

# Forwarded to each inference_fewshot.sh invocation, e.g.
# ./scripts/inference_fewshot_loop.sh --modes static_diverse --staged indo sun min
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
batch_langs=()

run_language() {
    local language="$1" gpu="$2" log_file="$3"
    CUDA_VISIBLE_DEVICES="$gpu" ./scripts/inference_fewshot.sh "$language" "${FEWSHOT_ARGS[@]}" > "$log_file" 2>&1
}

wait_batch() {
    local status=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "Finished language: ${batch_langs[$i]}"
        else
            echo "FAILED language: ${batch_langs[$i]} (see logs/fewshot_${batch_langs[$i]}.log)" >&2
            status=1
        fi
    done
    pids=()
    batch_langs=()
    return "$status"
}

idx=0
overall_status=0
for language in "${languages[@]}"; do
    gpu="${GPU_IDS[$((idx % NUM_GPUS))]}"
    log_file="logs/fewshot_${language}.log"
    echo "Running few-shot inference for language: $language on GPU $gpu (log: $log_file)"
    run_language "$language" "$gpu" "$log_file" &
    pids+=($!)
    batch_langs+=("$language")
    idx=$((idx + 1))

    if (( idx % NUM_GPUS == 0 )); then
        wait_batch || overall_status=1
    fi
done
wait_batch || overall_status=1

echo "----------------------------------------"
if [[ $overall_status -eq 0 ]]; then
    echo "All language few-shot inferences completed."
else
    echo "Some language few-shot inferences FAILED — check logs/ above." >&2
fi
exit "$overall_status"
