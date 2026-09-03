#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# vLLM's default worker subprocess start method is "fork". The "dynamic"
# few-shot condition loads a second model (Qwen3-Embedding-0.6B via
# sentence-transformers, on CPU) in-process before the vLLM engine starts;
# that load spins up background threads (tokenizer/OMP thread pools), and
# forking a process with live threads holding locks can deadlock the child
# the moment it touches an inherited-but-frozen lock. Force "spawn" so the
# EngineCore worker starts from a clean interpreter instead of forking.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Usage: ./scripts/inference_fewshot.sh [language] [--modes "mode1 mode2"] [--staged]
# Example: ./scripts/inference_fewshot.sh indo --modes static_diverse --staged
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

# Qwen3-8B's weights alone take ~15.3 GiB in bf16; on a single ~22.5 GiB
# GPU (e.g. an L4) the default gpu_memory_utilization=0.55 (~12.4 GiB
# budget) doesn't even cover the weights, let alone KV cache. Override
# per-hardware via GPU_MEMORY_UTILIZATION, e.g.
# GPU_MEMORY_UTILIZATION=0.7 ./scripts/inference_fewshot.sh indo
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# At gpu_memory_utilization=0.9 on a ~22.5 GiB GPU, Qwen3-8B's native
# 40960-token max_model_len needs more KV cache than fits. This runner is
# single-pass (max ~8704 generation tokens) with short hotel-review inputs,
# so cap it well below vLLM's reported ceiling. Override via MAX_MODEL_LEN.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Pass --modes "mode1 mode2" to run only a subset of conditions, e.g.
# ./scripts/inference_fewshot.sh indo --modes static_diverse
read -ra MODES <<< "${MODES_ARG:-zero static_curated static_random static_diverse dynamic}"

for seed in 123 777 2024 9584 31415; do
    for mode in "${MODES[@]}"; do
        python -m src.main.run_agent_fewshot \
            --model_path Qwen/Qwen3-0.6B \
            --test_case_path dataset/hotel_reviews/$language/mvp_aos/test.json \
            --prompt_set "${MODE_PROMPTSET[$mode]}" \
            --fewshot_mode "$mode" \
            --num_fewshot 3 \
            --fewshot_seed 42 \
            --embedding_model_name Qwen/Qwen3-Embedding-0.6B \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --seed $seed \
            --track_tokens \
            $STAGED_FLAG
    done
done
