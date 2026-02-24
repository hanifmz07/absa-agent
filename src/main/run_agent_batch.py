import argparse
import json
import time
import logging
import os
from typing import List, Dict

from ..utils.agent_batch import BatchABSASystem
from ..utils.parsing import parse_absa_string, convert_to_absa_format

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    model_path = "Qwen/Qwen3-0.6B"
    logger.info(f"Initializing Batch ABSA System with {model_path}...")
    system = BatchABSASystem(model_path, prompts_dir=f"prompts/{args.prompt_set}/", seed=args.seed)

    # Load data
    logger.info(f"Loading test cases from: {args.test_case_path}")
    with open(args.test_case_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Take first N test cases for quick testing
    test_cases = test_cases[:5]

    # Remove '[A] [O] [S]' from input text and build dataset format expected by BatchABSASystem
    dataset = []
    targets = {}
    for case in test_cases:
        input_text = case["input"].replace("[A] [O] [S]", "").strip()
        dataset.append({"id": case["sentence_id"], "text": input_text})
        targets[case["sentence_id"]] = parse_absa_string(case["target"])

    logger.info(
        f"Starting benchmark on {len(dataset)} test cases in batch mode "
        f"(max_retries={args.max_retries})"
    )

    overall_start = time.time()
    batch_results = system.process_dataset(dataset, max_retries=args.max_retries)
    overall_elapsed = time.time() - overall_start

    logger.info(f"All cases finished in {overall_elapsed:.2f}s total processing time")

    # Merge targets into results
    results = [
        {
            "id": item["id"],
            "input_text": item["text"],
            "target_text": targets[item["id"]],
            "status": item["status"],
            "attempts": item["attempts"],
            "final_extraction": item["final_output"],
            "history": item["history"],
        }
        for item in batch_results
    ]


    # Save to dir
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/batch"
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
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_batch.log")

    parser = argparse.ArgumentParser(
        description="Run the batch ABSA agent on a test dataset."
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
