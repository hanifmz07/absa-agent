#!/bin/bash

set -euo pipefail

OUTPUT_DIR="${1:-}"
EXP_SCHEME="${2:-async}"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_all_exact_match.sh <output_dir> [exp_scheme]"
    exit 1
fi


mapfile -t INFERENCE_PATHS < <(find "$OUTPUT_DIR/$EXP_SCHEME" -type f -name "inference_results.json" | sort)

if [ ${#INFERENCE_PATHS[@]} -eq 0 ]; then
    echo "No inference_results.json files found under: $OUTPUT_DIR/$EXP_SCHEME"
    exit 0
fi

echo "Found ${#INFERENCE_PATHS[@]} file(s)."

for INFERENCE_PATH in "${INFERENCE_PATHS[@]}"; do
    MODEL_NAME=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-8)}')
    DATASET_TYPE=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-7)}')
    LANG=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-6)}')
    DATASET_FOLDER=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-5)}')
    PROMPT_DIR=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-4)}')
    MAX_RETRIES=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-3)}')
    SEED=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-2)}')
    RUN_ID=$(echo "$INFERENCE_PATH" | awk -F/ '{print $(NF-1)}')

    echo "========================================================"
    echo "Running exact-match eval for:"
    echo "model=$MODEL_NAME dataset_type=$DATASET_TYPE lang=$LANG dataset_folder=$DATASET_FOLDER prompt_dir=$PROMPT_DIR max_retries=$MAX_RETRIES seed=$SEED run_id=$RUN_ID"
    echo "========================================================"

    bash scripts/eval_exact_match.sh \
        "$OUTPUT_DIR" \
        "$EXP_SCHEME" \
        "$MODEL_NAME" \
        "$DATASET_TYPE" \
        "$LANG" \
        "$DATASET_FOLDER" \
        "$PROMPT_DIR" \
        "$MAX_RETRIES" \
        "$SEED" \
        "$RUN_ID"
done
