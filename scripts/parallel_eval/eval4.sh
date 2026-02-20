#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=6

LANGUAGE="$1"
if [ -z "$LANGUAGE" ]; then
    echo "Error: No language specified."
    exit 1
fi
# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" && "$LANGUAGE" != "jav" && "$LANGUAGE" != "mad" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', 'sunda', 'jav', or 'mad'."
    exit 1
fi

DATASET_TYPE="$2"
# Validate the dataset type argument
if [ -z "$DATASET_TYPE" ]; then
    echo "Error: Dataset type must be specified."
    exit 1 # Exit with a non-zero status to indicate an error
fi

DATASET_FOLDER="$3"
# Validate the dataset folder argument
if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder must be specified. Name a folder located in the hotel_dataset/{lang} directory."
    exit 1 # Exit with a non-zero status to indicate an error
fi

BATCH_SIZE="$4"
# Validate the batch size argument
if [ -z "$BATCH_SIZE" ]; then
    echo "Error: Batch size must be specified."
    exit 1 # Exit with a non-zero status to indicate an error
fi


# Seeds for the SFT process
SEED=31415

# Define the log file names for clarity
LOG_BASE_NAME="eval"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}"

echo "========================================================" > "$STDOUT_LOG"
echo "Starting Evaluation script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

>"$STDERR_LOG"

{
  TEST_JSON="dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/test.json"
  MODEL_DIR="outputs/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/"
  OUTPUT_DIR="outputs/evals/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/"

  # Get all models in the directory and evaluate them one by one
  MODEL_PATHS="$MODEL_DIR/*"

  for MODEL_PATH in $MODEL_PATHS; do
    # Get all checkpoint paths within the model directory
    CHECKPOINT_PATHS="$MODEL_PATH/checkpoint-*"
    # Get the most recent checkpoint based on modification time
    LATEST_CHECKPOINT=$(ls -td $CHECKPOINT_PATHS | head -n 1)

    MODEL_NAME=$(basename "$MODEL_PATH")
    CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")

    EVAL_RESULT_DIR="$OUTPUT_DIR/$MODEL_NAME/$CHECKPOINT_NAME"

    # If the dataset folder is "mvp_aos" or "mvp_aos_augment", use the "mvp" prompt type, if gas use "gas", else use "indolegoabsa"
    if [ "$DATASET_FOLDER" == "mvp_aos" ] || [ "$DATASET_FOLDER" == "mvp_aos_augment" ]; then
      PROMPT_TYPE="mvp"
    elif [ "$DATASET_FOLDER" == "gas" ]; then
      PROMPT_TYPE="gas"
    elif [ "$DATASET_FOLDER" == "legoabsa_multitask" ] || [ "$DATASET_FOLDER" == "legoabsa_tasktransfer" ] || [ "$DATASET_FOLDER" == "indolegoabsa_multitask" ]; then
      PROMPT_TYPE="legoabsa"
    else
      # Error handling for unexpected dataset folder
      echo "Error: Unexpected dataset folder '$DATASET_FOLDER'. Cannot determine prompt type."
      exit 1
    fi

    # Only run if model folder exists and hasn't been evaluated yet
    if [ -d "$MODEL_PATH" ] && [ ! -d "$EVAL_RESULT_DIR/constrained_decoding" ] && [ ! -d "$EVAL_RESULT_DIR/unconstrained_decoding" ]; then
      echo "-----------------------------------------------------------"
      echo "Evaluating model: $MODEL_NAME (Seed: $SEED)"
      echo "-----------------------------------------------------------"
      
      python -m src.main.eval \
        --test_json_path "$TEST_JSON" \
        --model_path "$LATEST_CHECKPOINT" \
        --prompt_type "$PROMPT_TYPE" \
        --output_dir "$EVAL_RESULT_DIR" \
        --batch_size $BATCH_SIZE \
        --save_predictions \
        --use_constrained_decoding

    else
      echo "-----------------------------------------------------------"
      echo "Skipping $MODEL_NAME — either not trained yet or already evaluated."
      echo "-----------------------------------------------------------"
    fi
  done

  echo ""
  echo "========================================================"
  echo "All evaluations completed at: $(date)"
  echo "========================================================"

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"
