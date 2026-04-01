"""Entry point for running the OpenRouter-based ABSA agent.

This script mirrors :mod:`run_agent_async` but uses
:class:`~src.utils.agent_openrouter.OpenRouterABSASystem` instead of the
local vLLM engine.  All concurrency, retry, and result-saving logic is
identical.

Usage example::

    export OPENROUTER_API_KEY="sk-or-..."
    python -m src.main.run_agent_openrouter \\
        --model "qwen/qwen3-8b" \\
        --test_case_path dataset/hoasa_hotel/indo/mvp_aos/test.json \\
        --max_retries 3 \\
        --prompt_set exp1

The results are stored under::

    results/openrouter/<prompt_set>/max_retries_<n>/<model_slug>/<timestamp>/
        inference_results.json
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Dict

from ..utils.agent_openrouter import OpenRouterABSASystem
from ..utils.parsing import parse_absa_string

logger = logging.getLogger(__name__)


async def process_single_case(
    system: OpenRouterABSASystem,
    case: Dict,
    max_retries: int = 3,
) -> Dict:
    """Run the agent loop for one test case and return a structured result.

    Args:
        system: Initialised :class:`OpenRouterABSASystem` instance.
        case: A single test-case dict with ``sentence_id``, ``input``, and
            ``target`` keys.
        max_retries: Maximum extraction attempts for this case.

    Returns:
        A dict suitable for serialisation to the output JSON file.
    """
    logger.debug(f"Starting Case ID {case['sentence_id']}")
    start_time = time.time()

    output = await system.process_review(
        case["sentence_id"], case["input"], max_retries=max_retries
    )

    elapsed = time.time() - start_time
    logger.debug(f"Finished Case ID {case['sentence_id']} in {elapsed:.2f}s")

    parsed_target = parse_absa_string(case["target"])

    return {
        "id": case["sentence_id"],
        "input_text": case["input"],
        "target_text": parsed_target,
        "status": output["status"],
        "attempts": output["attempts"],
        "final_extraction": output["final_output"],
        "history": output["history"],
    }


async def main(args: argparse.Namespace) -> None:
    logger.info(
        f"Initialising OpenRouter ABSA System "
        f"(model={args.model}, prompt_set={args.prompt_set})"
    )
    system = OpenRouterABSASystem(
        model_name=args.model,
        prompts_dir=f"prompts/{args.prompt_set}/",
        api_key=args.api_key or None,
        site_url=args.site_url or None,
        site_name=args.site_name or None,
    )

    # Load test cases
    logger.info(f"Loading test cases from: {args.test_case_path}")
    with open(args.test_case_path, "r", encoding="utf-8") as fh:
        test_cases = json.load(fh)

    # Strip the task-format prefix that some datasets include in the input
    for case in test_cases:
        case["input"] = case["input"].replace("[A] [O] [S]", "").strip()

    logger.info(
        f"Starting benchmark on {len(test_cases)} test cases concurrently "
        f"(max_retries={args.max_retries})"
    )

    overall_start = time.time()

    tasks = [
        process_single_case(system, case, max_retries=args.max_retries)
        for case in test_cases
    ]
    results = await asyncio.gather(*tasks)

    overall_elapsed = time.time() - overall_start
    logger.info(f"All cases finished in {overall_elapsed:.2f}s total")

    # Build output directory: results/openrouter/<prompt_set>/max_retries_<n>/<model_slug>/<timestamp>/
    # Replace '/' in the model name with '__' so it is a valid directory name.
    model_slug = args.model.replace("/", "__")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        "results",
        "openrouter",
        args.prompt_set,
        f"max_retries_{args.max_retries}",
        model_slug,
        timestamp,
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "inference_results.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(list(results), fh, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    from logging_config import setup_logging

    os.makedirs("logs", exist_ok=True)
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_openrouter.log")

    parser = argparse.ArgumentParser(
        description="Run the OpenRouter ABSA agent on a test dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-8b",
        help="OpenRouter model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--test_case_path",
        type=str,
        default="dataset/hoasa_hotel/indo/mvp_aos/test.json",
        help="Path to the JSON test-case file (default: %(default)s).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum extraction attempts per review (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        default="exp1",
        help="Subdirectory inside prompts/ to load templates from (default: %(default)s).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "OpenRouter API key.  Defaults to the OPENROUTER_API_KEY "
            "environment variable when not provided."
        ),
    )
    parser.add_argument(
        "--site_url",
        type=str,
        default=None,
        help="Optional HTTP-Referer header value sent to OpenRouter.",
    )
    parser.add_argument(
        "--site_name",
        type=str,
        default=None,
        help="Optional X-Title header value sent to OpenRouter.",
    )

    parsed_args = parser.parse_args()
    asyncio.run(main(parsed_args))
