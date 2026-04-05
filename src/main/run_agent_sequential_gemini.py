import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

from ..utils.agent_sequential_gemini import SequentialGeminiABSASystem
from ..utils.parsing import parse_absa_string

logger = logging.getLogger(__name__)


def _extract_dataset_parts(test_case_path: str) -> Tuple[str, str, str]:
    # Normalize separators so Linux can also parse Windows-style paths.
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


def main(args: argparse.Namespace) -> None:
    logger.info("Initializing Sequential Gemini ABSA System with %s...", args.model_name)
    _, lang, _ = _extract_dataset_parts(args.test_case_path)
    system = SequentialGeminiABSASystem(
        model_name=args.model_name,
        prompts_dir=f"prompts/{args.prompt_set}/",
        track_tokens=args.track_tokens,
        api_key=args.gemini_api_key,
        max_api_retries=args.max_api_retries,
        retry_base_sleep_seconds=args.retry_base_sleep_seconds,
    )
    system.set_language_from_code(lang)

    logger.info("Loading test cases from: %s", args.test_case_path)
    with open(args.test_case_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    # Optionally limit the number of test cases for quicker runs
    start_idx = 800
    end_idx = len(test_cases)
    end_idx = 1000
    test_cases = test_cases[start_idx:end_idx]

    if args.limit is not None:
        test_cases = test_cases[: args.limit]

    for case in test_cases:
        case["input"] = case["input"].replace("[A] [O] [S]", "").strip()

    logger.info(
        "Starting sequential benchmark on %s test cases (max_retries=%s)",
        len(test_cases),
        args.max_retries,
    )

    overall_start = time.time()
    results = []

    for idx, case in enumerate(test_cases, start=1):
        logger.info("[%s/%s] Processing Case ID %s", idx, len(test_cases), case["sentence_id"])
        start_time = time.time()

        output = system.process_review(
            case["sentence_id"],
            case["input"],
            max_retries=args.max_retries,
        )

        elapsed = time.time() - start_time
        logger.info("Case ID %s finished in %.2fs", case["sentence_id"], elapsed)

        parsed_target = parse_absa_string(case["target"])
        results.append(
            {
                "id": case["sentence_id"],
                "input_text": case["input"],
                "target_text": parsed_target,
                "status": output["status"],
                "attempts": output["attempts"],
                "final_extraction": output["final_output"],
                "history": output["history"],
                "token_usage": output.get("token_usage"),
            }
        )

    overall_elapsed = time.time() - overall_start
    logger.info("All cases finished in %.2fs total processing time", overall_elapsed)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    dataset_type, lang, dataset_folder = _extract_dataset_parts(args.test_case_path)

    output_dir = Path("results") / "sequential_gemini" / f"limit_{start_idx}_{end_idx}"
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
    setup_logging(level=logging.INFO, log_file="logs/absa_agent_sequential_gemini.log")

    parser = argparse.ArgumentParser(
        description="Run the sequential Gemini ABSA agent on a test dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3-flash-preview",
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
        help="Enable per-instance token counting (input, thinking, output). Counts are saved under 'token_usage'.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test cases to run from the start of the dataset.",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
