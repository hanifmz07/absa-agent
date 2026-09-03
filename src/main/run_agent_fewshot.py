import argparse
import json
import time
import asyncio
import logging
from typing import Dict
import os

from ..utils.agent_fewshot import FewShotABSASystem
from ..utils.fewshot_pool import (
    load_pool,
    sample_static_random,
    sample_diverse_curated,
    build_embedding_index,
    retrieve_topk,
    format_fewshot_block,
)
from ..utils.parsing import parse_absa_string

logger = logging.getLogger(__name__)

# Which --prompt_set directory each (--fewshot_mode, --staged) pairing is
# expected to load from. static_curated reuses prompts/exp1/ (non-staged) or
# prompts/exp2_staged/ (staged) unchanged, each with its own baked-in
# examples; the other three modes load one of the new fewshot_*[/_staged]
# prompt sets, which differ from their sibling only in whether the system
# prompt asks for staged 3-stage reasoning and whether injected examples
# carry a synthesized reasoning trace (see FEWSHOT_EXPERIMENT_PLAN.md).
# Enforced in main() so a mismatched pairing (e.g. --fewshot_mode dynamic
# --prompt_set exp1, which would silently run as zero-shot since exp1's
# template ignores {fewshot_examples}) fails loudly instead of producing a
# silently mislabeled results directory.
MODE_PROMPTSET = {
    ("zero", False): "fewshot_zeroshot",
    ("zero", True): "fewshot_zeroshot_staged",
    ("static_curated", False): "exp1",
    ("static_curated", True): "exp2_staged",
    ("static_random", False): "fewshot_static_random",
    ("static_random", True): "fewshot_static_random_staged",
    ("static_diverse", False): "fewshot_static_diverse",
    ("static_diverse", True): "fewshot_static_diverse_staged",
    ("dynamic", False): "fewshot_dynamic",
    ("dynamic", True): "fewshot_dynamic_staged",
}


def _extract_dataset_parts(test_case_path: str):
    normalized = test_case_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]

    try:
        dataset_index = parts.index("dataset")
    except ValueError as exc:
        raise ValueError(
            "test_case_path must contain 'dataset/<dataset_type>/<lang>/<dataset_folder>/...'"
        ) from exc

    if len(parts) <= dataset_index + 3:
        raise ValueError(
            "test_case_path must follow dataset/<dataset_type>/<lang>/<dataset_folder>/..."
        )

    return parts[dataset_index + 1], parts[dataset_index + 2], parts[dataset_index + 3]


async def process_single_case(
    system: FewShotABSASystem,
    case: Dict,
    fewshot_block: str = "",
) -> Dict:
    """Helper function to run a single case and track its timing concurrently."""
    logger.debug(f"Starting Case ID {case['sentence_id']}")
    start_time = time.time()

    output = await system.process_review(
        case['sentence_id'], case['input'], fewshot_block=fewshot_block
    )

    elapsed = time.time() - start_time
    logger.debug(f"Finished Case ID {case['sentence_id']} in {elapsed:.2f}s")

    parsed_target = parse_absa_string(case['target'])

    return {
        'id': case['sentence_id'],
        'input_text': case['input'],
        'target_text': parsed_target,
        'status': output['status'],
        'attempts': output['attempts'],
        'final_extraction': output['final_output'],
        'history': output['history'],
        'token_usage': output.get('token_usage'),
    }


async def main(args: argparse.Namespace):
    expected_prompt_set = MODE_PROMPTSET[(args.fewshot_mode, args.staged)]
    if args.prompt_set != expected_prompt_set:
        raise ValueError(
            f"--fewshot_mode {args.fewshot_mode!r} with --staged={args.staged} expects "
            f"--prompt_set {expected_prompt_set!r}, got {args.prompt_set!r}"
        )

    model_path = args.model_path
    model_name = model_path.split('/')[-1]
    dataset_type, lang, dataset_folder = _extract_dataset_parts(args.test_case_path)

    # Build few-shot resources BEFORE the vLLM engine: for `dynamic` mode
    # the embedding model + pool embeddings should be loaded and computed
    # ahead of AsyncLLMEngine construction, since vLLM's
    # gpu_memory_utilization profiling is sensitive to memory already
    # allocated at that point.
    static_block = ""
    pool = None
    embed_model = None
    pool_embeddings = None

    if args.fewshot_mode == "static_random":
        logger.info(
            f"Sampling {args.num_fewshot} static-random few-shot examples "
            f"(seed={args.fewshot_seed}) from {dataset_type}/{lang}/{dataset_folder}/train.json"
        )
        pool = load_pool(dataset_type, lang, dataset_folder)
        examples = sample_static_random(pool, args.num_fewshot, args.fewshot_seed)
        static_block = format_fewshot_block(examples, staged=args.staged)
    elif args.fewshot_mode == "static_diverse":
        logger.info(
            f"Sampling 5 diverse-curated few-shot examples (seed={args.fewshot_seed}) "
            f"from {dataset_type}/{lang}/{dataset_folder}/train.json"
        )
        pool = load_pool(dataset_type, lang, dataset_folder)
        examples = sample_diverse_curated(pool, args.fewshot_seed)
        static_block = format_fewshot_block(examples, staged=args.staged)
    elif args.fewshot_mode == "dynamic":
        logger.info(
            f"Building embedding index for dynamic retrieval from "
            f"{dataset_type}/{lang}/{dataset_folder}/train.json"
        )
        pool = load_pool(dataset_type, lang, dataset_folder)
        embed_model, pool_embeddings = build_embedding_index(pool, args.embedding_model_name)

    logger.info(f"Initializing FewShot ABSA System with {model_path}...")
    system = FewShotABSASystem(
        model_path,
        prompts_dir=f"prompts/{args.prompt_set}/",
        seed=args.seed,
        track_tokens=args.track_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    system.set_language_from_code(lang)

    logger.info(f"Loading test cases from: {args.test_case_path}")
    with open(args.test_case_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    for case in test_cases:
        case['input'] = case['input'].replace('[A] [O] [S]', '').strip()

    def fewshot_for(case: Dict) -> str:
        if args.fewshot_mode in ("static_random", "static_diverse"):
            return static_block
        if args.fewshot_mode == "dynamic":
            examples = retrieve_topk(case['input'], embed_model, pool, pool_embeddings, k=args.num_fewshot)
            return format_fewshot_block(examples, staged=args.staged)
        return ""  # zero, static_curated

    logger.info(
        f"Starting single-pass benchmark on {len(test_cases)} test cases concurrently "
        f"(fewshot_mode={args.fewshot_mode})"
    )

    overall_start = time.time()

    tasks = [
        process_single_case(system, case, fewshot_block=fewshot_for(case))
        for case in test_cases
    ]

    results = await asyncio.gather(*tasks)

    overall_elapsed = time.time() - overall_start
    logger.info(f"All cases finished in {overall_elapsed:.2f}s total processing time")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "results/fewshot_single_pass"
    output_dir = os.path.join(output_dir, model_name)
    output_dir = os.path.join(output_dir, dataset_type)
    output_dir = os.path.join(output_dir, lang)
    output_dir = os.path.join(output_dir, dataset_folder)
    output_dir = os.path.join(output_dir, args.prompt_set)
    # Not a real retry count -- there is exactly one extractor call per
    # item. Kept as a literal path segment so eval scripts' positional
    # metadata parsing (extract_metadata() in eval_exact_match.py /
    # eval_instruct_absa.py / eval_semantic.py) keeps working unmodified.
    output_dir = os.path.join(output_dir, "max_retries_1")
    output_dir = os.path.join(output_dir, f"seed_{args.seed}")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'inference_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {os.path.join(output_dir, 'inference_results.json')}")


if __name__ == "__main__":
    from logging_config import setup_logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_fewshot.log")

    parser = argparse.ArgumentParser(
        description="Run the single-pass few-shot ABSA agent on a test dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path/url to the pre-trained model (default: %(default)s).",
    )
    parser.add_argument(
        "--test_case_path",
        type=str,
        default="dataset/hotel_reviews/indo/mvp_aos/test.json",
        help="Path to the JSON test-case file (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        default="fewshot_zeroshot",
        help="Prompt set to use for the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--fewshot_mode",
        type=str,
        choices=["zero", "static_curated", "static_random", "static_diverse", "dynamic"],
        default="zero",
        help="Few-shot example condition (default: %(default)s).",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=3,
        help="Number of few-shot examples for static_random/dynamic modes (default: %(default)s). "
             "Ignored for static_diverse, which always selects exactly 5 (one per category).",
    )
    parser.add_argument(
        "--fewshot_seed",
        type=int,
        default=42,
        help="Seed for static_random example sampling, independent of --seed "
             "so the example set stays fixed across a generation-seed sweep (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model for dynamic retrieval (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the vLLM engine (default: %(default)s).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.55,
        help="Fraction of GPU memory vLLM is allowed to use (default: %(default)s, "
             "matching exp1's agent_async.py). Raise this if you see "
             "'No available memory for the cache blocks' on a smaller/otherwise-idle GPU.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="Max sequence length (prompt + generation) reserved for the vLLM KV cache "
             "(default: %(default)s, comfortably covers this runner's single-pass prompt "
             "+ ~8704-token generation budget). Raise --gpu_memory_utilization or "
             "lower this if you still see a KV-cache-too-small error.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        default=False,
        help="Use the staged 3-stage reasoning-process prompt variant "
             "(sentiment markers -> related aspects -> polarity classification) "
             "instead of the plain fewshot prompt, and inject a matching "
             "synthesized reasoning trace into static_random/dynamic example "
             "blocks. Requires --prompt_set to be the '_staged' (or, for "
             "static_curated, 'exp2_staged') counterpart (default: %(default)s).",
    )
    parser.add_argument(
        "--track_tokens",
        action="store_true",
        default=False,
        help="Enable per-instance token counting (input, thinking, output). "
             "Counts are saved under 'token_usage' in the results file.",
    )
    parsed_args = parser.parse_args()

    asyncio.run(main(parsed_args))
