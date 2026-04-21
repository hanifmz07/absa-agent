#!/bin/bash

set -euo pipefail

source .venv/bin/activate

OUTPUT_DIR="${1:-}"
EXP_SCHEME="${2:-async}"
MODEL_NAME="${3:-*}"
DATASET_TYPE="${4:-*}"
LANG="${5:-*}"
DATASET_FOLDER="${6:-*}"
PROMPT_DIR="${7:-*}"
MAX_RETRIES="${8:-*}"
SEED="${9:-*}"
RUN_ID="${10:-*}"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: output_dir is required."
    echo "Usage: bash scripts/eval_instruct_absa.sh <output_dir> [exp_scheme=async] [model_name] [dataset_type] [lang] [dataset_folder] [prompt_dir] [max_retries] [seed] [run_id]"
    exit 1
fi

INFERENCE_GLOB="$OUTPUT_DIR/$EXP_SCHEME/$MODEL_NAME/$DATASET_TYPE/$LANG/$DATASET_FOLDER/$PROMPT_DIR/$MAX_RETRIES/$SEED/$RUN_ID/inference_results.json"

echo "Running instruct_absa eval for pattern: $INFERENCE_GLOB"
python -m src.main.eval_instruct_absa --inference_path "$INFERENCE_GLOB"
