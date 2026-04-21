#!/bin/bash

set -euo pipefail

source .venv/bin/activate

OUTPUT_DIR="${1:-}"
EMBEDDING_MODEL_NAME="${2:-}"
EXP_SCHEME="${3:-async}"
MODEL_NAME="${4:-*}"
DATASET_TYPE="${5:-*}"
LANG="${6:-*}"
DATASET_FOLDER="${7:-*}"
PROMPT_DIR="${8:-*}"
MAX_RETRIES="${9:-*}"
SEED="${10:-*}"
RUN_ID="${11:-*}"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_semantic.sh <output_dir> <embedding_model_name> [exp_scheme=async] [model_name] [dataset_type] [lang] [dataset_folder] [prompt_dir] [max_retries] [seed] [run_id]"
    exit 1
fi

if [ -z "$EMBEDDING_MODEL_NAME" ]; then
    echo "Error: embedding_model_name is required."
    echo "Usage: bash scripts/eval_semantic.sh <output_dir> <embedding_model_name> [exp_scheme=async] [model_name] [dataset_type] [lang] [dataset_folder] [prompt_dir] [max_retries] [seed] [run_id]"
    exit 1
fi

INFERENCE_GLOB="$OUTPUT_DIR/$EXP_SCHEME/$MODEL_NAME/$DATASET_TYPE/$LANG/$DATASET_FOLDER/$PROMPT_DIR/$MAX_RETRIES/$SEED/$RUN_ID/inference_results.json"

echo "Running semantic eval for pattern: $INFERENCE_GLOB"
python -m src.main.eval_semantic \
    --inference_path "$INFERENCE_GLOB" \
    --embedding_model_name "$EMBEDDING_MODEL_NAME"
