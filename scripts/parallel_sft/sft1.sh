#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

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
SEED=9584

# Define the log file names for clarity
LOG_BASE_NAME="sft_full"
LOG_DIR="logs"
PID=$$
STDOUT_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/${PID}_$(date).log"
STDERR_LOG="${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}/${PID}_$(date).err"
# Create necessary directories for logs
mkdir -p "${LOG_DIR}/${LOG_BASE_NAME}/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_${SEED}"

# --- Step 1: Initialize Log Files ---
# Clear previous logs and add a timestamp to mark the start of this run.
# The '>' operator truncates the file to zero before writing.
echo "========================================================" > "$STDOUT_LOG"
echo "Starting SFT script run at: $(date)" >> "$STDOUT_LOG"
echo "========================================================" >> "$STDOUT_LOG"

# We do the same for the error log, in case of any setup errors.
>"$STDERR_LOG"

# --- Step 2: Group the entire process and apply redirection ---
# The curly braces { ... } group all commands inside.
# The redirection at the end applies to everything within the braces,
# including all the 'echo' statements and every python run.
# We use 'tee -a' to append to our initialized logs.
{
    echo "Running ABSA SFT"

    echo ""
    echo "--------------------------------------------------------"
    echo "Running full sft with seed: $SEED"
    echo "--------------------------------------------------------"

    # DATASET_FOLDERS=("indolegoabsa_multitask" "legoabsa_multitask" "legoabsa_tasktransfer" "mvp_aos" "gas")
    DATASET_FOLDERS=("indolegoabsa_multitask" "legoabsa_multitask" "legoabsa_tasktransfer" "mvp_aos" "gas")
    for DATASET_FOLDER in "${DATASET_FOLDERS[@]}"; do
        echo "Processing dataset folder: $DATASET_FOLDER"

        if [ "$DATASET_FOLDER" == "mvp_aos" ] || [ "$DATASET_FOLDER" == "mvp_aos_augment" ]; then
            PROMPT_TYPE="mvp"
        elif [ "$DATASET_FOLDER" == "gas" ]; then
            PROMPT_TYPE="gas"
        elif [ "$DATASET_FOLDER" == "legoabsa_multitask" ] || [ "$DATASET_FOLDER" == "legoabsa_tasktransfer" ] || [ "$DATASET_FOLDER" == "indolegoabsa_multitask" ]; then
            PROMPT_TYPE="legoabsa"
        else
            echo "Unknown dataset folder: $DATASET_FOLDER"
            continue
        fi

        # The output of this python command will now be correctly
        # redirected along with everything else in the loop.
        python -m src.main.train \
            --train_json_path "dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/train.json" \
            --model_name "Qwen/Qwen2.5-0.5B" \
            --output_dir "outputs/models/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/seed_$SEED/" \
            --prompt_type "$PROMPT_TYPE" \
            --save_strategy "epoch" \
            --num_epochs 10 \
            --lr 5e-5 \
            --optimizer "adamw_torch" \
            --seed $SEED \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --eval_strategy "epoch" \
            --val_json_path "dataset/${DATASET_TYPE}/${LANGUAGE}/${DATASET_FOLDER}/dev.json" \
            --val_batch_size 16
        echo ""
        echo "========================================================"
        echo "All seeds completed at: $(date)"
        echo "========================================================"
    done

} 2> >(tee "$STDERR_LOG") | tee "$STDOUT_LOG"