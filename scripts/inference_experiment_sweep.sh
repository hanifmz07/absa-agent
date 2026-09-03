#!/usr/bin/env bash
# Run three experiment configs across all languages, 5-seed sweep each:
#   exp1                    -> run_agent_async  --prompt_set exp1        (evaluator/retry)
#   exp1_staged             -> run_agent_async  --prompt_set exp2_staged (evaluator/retry, staged)
#   fewshot_static_diverse  -> run_agent_fewshot --fewshot_mode static_diverse (single-pass)
#
# Runs sequentially -- every seed spins up its own vLLM engine.
#
# Overridable via env vars:
#   MODEL_PATH              (default Qwen/Qwen3-8B)
#   DATASET_TYPE            (default hotel_reviews)
#   DATASET_FOLDER          (default mvp_aos)
#   MAX_RETRIES             (default 10)   -- exp1 / exp1_staged only
#   GPU_MEMORY_UTILIZATION  (default 0.9)  -- all three configs
#   MAX_MODEL_LEN           (default 24576) -- all three configs (exp1/exp1_staged run an
#                                              evaluator/retry loop whose extractor prompt grows
#                                              with accumulated critique history across retries,
#                                              so this needs more headroom than fewshot's
#                                              single-pass prompt strictly requires)
#   SEEDS                   (default "123 777 2024 9584 31415")
#   LANGUAGES              (default "indo sun min jav mad eng")
#   CONFIGS                (default "exp1 exp1_staged fewshot_static_diverse")
#
# Examples:
#   ./scripts/inference_experiment_sweep.sh
#   CONFIGS=fewshot_static_diverse LANGUAGES="indo sun" ./scripts/inference_experiment_sweep.sh
#   MODEL_PATH=Qwen/Qwen3-0.6B SEEDS=42 LANGUAGES=indo CONFIGS=exp1 ./scripts/inference_experiment_sweep.sh
set -uo pipefail          # NOT -e: one failed run must not abort the whole sweep

source .venv/bin/activate

# vLLM's default worker start method is "fork"; force "spawn" so an EngineCore
# worker starts from a clean interpreter (matches scripts/inference_fewshot.sh).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATASET_TYPE="${DATASET_TYPE:-hotel_reviews}"
DATASET_FOLDER="${DATASET_FOLDER:-mvp_aos}"
MAX_RETRIES="${MAX_RETRIES:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"

read -ra SEEDS     <<< "${SEEDS:-123 777 2024 9584 31415}"
read -ra LANGUAGES <<< "${LANGUAGES:-indo sun min jav mad eng}"
read -ra CONFIGS   <<< "${CONFIGS:-exp1 exp1_staged fewshot_static_diverse}"

failures=()

test_case_path() {
    echo "dataset/${DATASET_TYPE}/$1/${DATASET_FOLDER}/test.json"
}

run_async() {   # $1 = language, $2 = prompt_set
    local language="$1" prompt_set="$2" seed
    for seed in "${SEEDS[@]}"; do
        echo "=== config=async prompt_set=${prompt_set} lang=${language} seed=${seed} ==="
        if ! python -m src.main.run_agent_async \
            --model_path "$MODEL_PATH" \
            --test_case_path "$(test_case_path "$language")" \
            --max_retries "$MAX_RETRIES" \
            --prompt_set "$prompt_set" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --seed "$seed"; then
            failures+=("${prompt_set}/${language}/seed_${seed}")
        fi
    done
}

run_fewshot_diverse() {   # $1 = language
    local language="$1" seed
    for seed in "${SEEDS[@]}"; do
        echo "=== config=fewshot_static_diverse lang=${language} seed=${seed} ==="
        if ! python -m src.main.run_agent_fewshot \
            --model_path "$MODEL_PATH" \
            --test_case_path "$(test_case_path "$language")" \
            --prompt_set fewshot_static_diverse \
            --fewshot_mode static_diverse \
            --fewshot_seed 42 \
            --embedding_model_name Qwen/Qwen3-Embedding-0.6B \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --seed "$seed" \
            --track_tokens; then
            failures+=("fewshot_static_diverse/${language}/seed_${seed}")
        fi
    done
}

for config in "${CONFIGS[@]}"; do
    for language in "${LANGUAGES[@]}"; do
        echo "----------------------------------------"
        echo "Running config=${config} language=${language}"
        echo "----------------------------------------"
        case "$config" in
            exp1)                   run_async "$language" exp1 ;;
            exp1_staged)            run_async "$language" exp2_staged ;;
            fewshot_static_diverse) run_fewshot_diverse "$language" ;;
            *)
                echo "Unknown config: ${config} (expected: exp1, exp1_staged, fewshot_static_diverse)" >&2
                exit 2
                ;;
        esac
    done
done

echo "========================================"
if [[ ${#failures[@]} -eq 0 ]]; then
    echo "All runs completed successfully."
    exit 0
fi

echo "The following runs FAILED (see logs/ above):" >&2
for f in "${failures[@]}"; do
    echo "  - $f" >&2
done
exit 1
