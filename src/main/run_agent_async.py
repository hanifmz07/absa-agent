import argparse
import json
import csv
import time
import asyncio
import logging
from typing import List, Dict
import os

from ..utils.agent_async import AsyncABSASystem 
from ..utils.parsing import parse_absa_string, convert_to_absa_format

logger = logging.getLogger(__name__)

async def process_single_case(
    system: AsyncABSASystem,
    case: Dict,
    max_retries: int = 3,
) -> Dict:
    """Helper function to run a single case and track its timing concurrently."""
    logger.debug(f"Starting Case ID {case['sentence_id']}")
    start_time = time.time()

    # Run the agent loop concurrently
    output = await system.process_review(
        case['sentence_id'], case['input'], max_retries=max_retries
    )
    
    elapsed = time.time() - start_time
    logger.debug(f"Finished Case ID {case['sentence_id']} in {elapsed:.2f}s")

    # Parse target for evaluation
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
    model_path = args.model_path
    model_name = model_path.split('/')[-1]
    logger.info(f"Initializing Async ABSA System with {model_path}...")
    system = AsyncABSASystem(model_path, prompts_dir=f"prompts/{args.prompt_set}/", seed=args.seed, track_tokens=args.track_tokens)

    # Load data
    logger.info(f"Loading test cases from: {args.test_case_path}")
    with open(args.test_case_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Take first 10 test cases for quick testing
    # test_cases = test_cases[:5]
    # error_cases_subtr = [3, 4, 7, 59, 85, 95, 98, 91]
    # test_cases = [test_cases[i] for i in error_cases_subtr]

    # Remove '[A] [O] [S]' from input text
    for case in test_cases:
        case['input'] = case['input'].replace('[A] [O] [S]', '').strip()

    logger.info(
        f"Starting benchmark on {len(test_cases)} test cases concurrently "
        f"(max_retries={args.max_retries})"
    )

    overall_start = time.time()

    # Create a list of async tasks, forwarding max_retries to each review.
    tasks = [
        process_single_case(system, case, max_retries=args.max_retries)
        for case in test_cases
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    overall_elapsed = time.time() - overall_start
    logger.info(f"All cases finished in {overall_elapsed:.2f}s total processing time")

    # Save to dir
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    dataset_type = args.test_case_path.split('/')[1]  # e.g., 'hotel_reviews'
    lang = args.test_case_path.split('/')[2]  # e.g., 'indo'
    dataset_folder = args.test_case_path.split('/')[3]  # e.g., 'mvp_aos'
    output_dir = f"results/async"
    output_dir = os.path.join(output_dir, model_name)
    output_dir = os.path.join(output_dir, dataset_type)
    output_dir = os.path.join(output_dir, lang)
    output_dir = os.path.join(output_dir, dataset_folder)
    output_dir = os.path.join(output_dir, args.prompt_set)
    output_dir = os.path.join(output_dir, f"max_retries_{args.max_retries}")
    output_dir = os.path.join(output_dir, f"seed_{args.seed}")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'inference_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Results saved to {os.path.join(output_dir, 'inference_results.json')}")

if __name__ == "__main__":
    from logging_config import setup_logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_async.log")

    parser = argparse.ArgumentParser(
        description="Run the async ABSA agent on a test dataset."
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
        "--max_retries",
        type=int,
        default=3,
        help="Maximum extraction attempts per review before giving up (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the vLLM engine (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        default="exp1",
        help="Prompt set to use for the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--track_tokens",
        action="store_true",
        default=False,
        help="Enable per-instance token counting (input, thinking, output). "
             "Counts are saved under 'token_usage' in the results file.",
    )
    parsed_args = parser.parse_args()

    # Launch the asyncio event loop
    asyncio.run(main(parsed_args))