import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

from ..utils.agent_async_gemini import AsyncGeminiABSASystem
from ..utils.parsing import parse_absa_string

logger = logging.getLogger(__name__)


async def process_single_case(
    system: AsyncGeminiABSASystem,
    case: Dict,
    max_retries: int = 3,
) -> Dict:
    logger.debug("Starting Case ID %s", case["sentence_id"])
    start_time = time.time()

    output = await system.process_review(
        case["sentence_id"], case["input"], max_retries=max_retries
    )

    elapsed = time.time() - start_time
    logger.debug("Finished Case ID %s in %.2fs", case["sentence_id"], elapsed)

    parsed_target = parse_absa_string(case["target"])

    return {
        "id": case["sentence_id"],
        "input_text": case["input"],
        "target_text": parsed_target,
        "status": output["status"],
        "attempts": output["attempts"],
        "final_extraction": output["final_output"],
        "history": output["history"],
        "token_usage": output.get("token_usage"),
    }


async def main(args: argparse.Namespace) -> None:
    logger.info("Initializing Async Gemini ABSA System with %s...", args.model_name)
    system = AsyncGeminiABSASystem(
        model_name=args.model_name,
        prompts_dir=f"prompts/{args.prompt_set}/",
        track_tokens=args.track_tokens,
        api_key=args.gemini_api_key,
        max_api_retries=args.max_api_retries,
        retry_base_sleep_seconds=args.retry_base_sleep_seconds,
    )

    logger.info("Loading test cases from: %s", args.test_case_path)
    with open(args.test_case_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    # Test on first 20 cases for quick iteration during development
    test_cases = test_cases[:20]

    for case in test_cases:
        case["input"] = case["input"].replace("[A] [O] [S]", "").strip()

    logger.info(
        "Starting benchmark on %s test cases concurrently (max_retries=%s)",
        len(test_cases),
        args.max_retries,
    )

    overall_start = time.time()
    tasks = [
        process_single_case(system, case, max_retries=args.max_retries)
        for case in test_cases
    ]
    results = await asyncio.gather(*tasks)

    overall_elapsed = time.time() - overall_start
    logger.info("All cases finished in %.2fs total processing time", overall_elapsed)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path_parts = Path(args.test_case_path).parts
    if len(path_parts) < 4:
        raise ValueError(
            "test_case_path must follow dataset/<dataset_type>/<lang>/<dataset_folder>/..."
        )

    dataset_type = path_parts[1]
    lang = path_parts[2]
    dataset_folder = path_parts[3]

    output_dir = Path("results") / "async_gemini"
    output_dir = output_dir / args.model_name
    output_dir = output_dir / dataset_type / lang / dataset_folder
    output_dir = output_dir / args.prompt_set
    output_dir = output_dir / f"max_retries_{args.max_retries}"
    output_dir = output_dir / f"seed_{args.seed}"
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "inference_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    from logging_config import setup_logging

    load_dotenv()

    os.makedirs("logs", exist_ok=True)
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_async_gemini.log")

    parser = argparse.ArgumentParser(
        description="Run the async Gemini ABSA agent on a test dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model id (default: %(default)s).",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key. If omitted, GEMINI_API_KEY env var is used.",
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
        help="Kept for output-folder parity with other runners (default: %(default)s).",
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
    parser.add_argument(
        "--max_api_retries",
        type=int,
        default=5,
        help="Maximum retries for retryable Gemini API errors like 429/quota (default: %(default)s).",
    )
    parser.add_argument(
        "--retry_base_sleep_seconds",
        type=float,
        default=2.0,
        help="Base sleep seconds for exponential backoff between API retries (default: %(default)s).",
    )
    parsed_args = parser.parse_args()

    asyncio.run(main(parsed_args))
