import argparse
import json
import csv
import time
import logging
from typing import List, Dict
from ..utils.agent_sequential import SequentialABSASystem 
from tqdm import tqdm
import os
from ..utils.parsing import parse_absa_string, convert_to_absa_format

logger = logging.getLogger(__name__)


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

def main(args: argparse.Namespace):
    model_path = "Qwen/Qwen3-0.6B" 
    _, lang, _ = _extract_dataset_parts(args.test_case_path)
    logger.info(f"Initializing ABSA System with {model_path}...")
    system = SequentialABSASystem(model_path, prompts_dir=f"prompts/{args.prompt_set}/", seed=args.seed)
    system.set_language_from_code(lang)

    # Load data
    logger.info(f"Loading test cases from: {args.test_case_path}")
    with open(args.test_case_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Take first 5 test cases for quick testing
    test_cases = test_cases[:5]

    # Remove '[A] [O] [S]' from input text
    for case in test_cases:
        case['input'] = case['input'].replace('[A] [O] [S]', '').strip()
    results = []

    logger.info(
        f"Starting benchmark on {len(test_cases)} test cases "
        f"(max_retries={args.max_retries})"
    )

    overall_start = time.time()

    # Run extractor-evaluator loop for each test case
    for case in tqdm(test_cases, desc="Processing test cases"):
        logger.debug(f"Processing Case ID {case['sentence_id']}")
        start_time = time.time()
        
        # Run agent loop
        output = system.process_review(case['input'], max_retries=args.max_retries)
        
        elapsed = time.time() - start_time
        logger.debug(f"Case {case['sentence_id']} finished in {elapsed:.2f}s")

        # Parse target for evaluation
        parsed_target = parse_absa_string(case['target'])

        results.append({
            'id': case['sentence_id'],
            'input_text': case['input'],
            'target_text': parsed_target,
            'status': output['status'],
            'attempts': output['attempts'],
            'final_extraction': output['final_output'],
            'history': output['history']
        })

    overall_elapsed = time.time() - overall_start
    logger.info(f"All cases finished in {overall_elapsed:.2f}s total processing time")


    # Save to dir
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/sequential"
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
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_sequential.log")

    parser = argparse.ArgumentParser(
        description="Run the sequential ABSA agent on a test dataset."
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
    parsed_args = parser.parse_args()

    main(parsed_args)