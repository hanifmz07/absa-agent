import argparse
import glob
import json
import os
from typing import Any, Dict, List

from ..utils.eval_utils import calculate_metrics, normalize_triplet_list
from ..utils.parsing import punctuation_triplets


def resolve_inference_paths(inference_path: str) -> List[str]:
    if os.path.isfile(inference_path):
        return [inference_path]

    matched_paths = sorted(glob.glob(inference_path, recursive=True))
    return [path for path in matched_paths if os.path.isfile(path)]


def extract_metadata(path: str) -> Dict[str, Any]:
    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    metadata: Dict[str, Any] = {
        "path": path,
        "runner_type": None,
        "model_name": None,
        "dataset_type": None,
        "lang": None,
        "dataset_folder": None,
        "prompt_dir": None,
        "max_retries": None,
        "seed": None,
        "run_id": None,
    }

    # Expected tail:
    # async/{model_name}/{dataset_type}/{lang}/{dataset_folder}/{prompt_dir}/{max_retries}/{seed}/{run_id}/inference_results.json
    if len(parts) >= 11:
        metadata.update(
            {
                "runner_type": parts[-10],
                "model_name": parts[-9],
                "dataset_type": parts[-8],
                "lang": parts[-7],
                "dataset_folder": parts[-6],
                "prompt_dir": parts[-5],
                "max_retries": parts[-4],
                "seed": parts[-3],
                "run_id": parts[-2],
            }
        )
    return metadata


def evaluate_file(inference_path: str) -> Dict[str, Any]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_lists = [
        punctuation_triplets(normalize_triplet_list(item.get("target_text", [])))
        for item in data
    ]
    prediction_lists = [
        punctuation_triplets(normalize_triplet_list(item.get("final_extraction", [])))
        for item in data
    ]
    scores = calculate_metrics(prediction_lists, target_lists, task="exact_match")

    result = extract_metadata(inference_path)
    result.update(
        {
            "precision": scores["precision_exact_match"] * 100,
            "recall": scores["recall_exact_match"] * 100,
            "f1": scores["f1_exact_match"] * 100,
        }
    )
    return result


def evaluate_file_details(inference_path: str) -> List[Dict[str, Any]]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    details: List[Dict[str, Any]] = []
    for item in data:
        target_list = punctuation_triplets(normalize_triplet_list(item.get("target_text", [])))
        prediction_list = punctuation_triplets(normalize_triplet_list(item.get("final_extraction", [])))
        score = calculate_metrics([prediction_list], [target_list], task="exact_match")
        false_positive = [pred for pred in prediction_list if pred not in target_list]
        false_negative = [target for target in target_list if target not in prediction_list]

        details.append(
            {
                "id": item.get("id"),
                "input_text": item.get("input_text"),
                "target_text": target_list,
                "final_extraction": prediction_list,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": score["precision_exact_match"] * 100,
                "recall": score["recall_exact_match"] * 100,
                "f1": score["f1_exact_match"] * 100,
            }
        )

    return details


def main(args: argparse.Namespace) -> None:
    inference_paths = resolve_inference_paths(args.inference_path)
    if not inference_paths:
        raise FileNotFoundError(f"No inference files found from: {args.inference_path}")

    results = []
    for inference_path in inference_paths:
        print(f"Processing: {inference_path}")
        summary_result = evaluate_file(inference_path)
        detail_result = evaluate_file_details(inference_path)
        results.append(summary_result)

        output_dir = os.path.dirname(inference_path)
        summary_path = os.path.join(output_dir, "exact_match.json")
        detail_path = os.path.join(output_dir, "exact_match_detail.json")

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        with open(detail_path, "w", encoding="utf-8") as file:
            json.dump(detail_result, file, indent=2, ensure_ascii=False)

        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved detail metrics to: {detail_path}")

    if len(results) == 1:
        print("Exact match metrics:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("Exact match metrics summary:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exact-match ABSA metrics from agent inference results")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path or glob pattern to inference_results.json",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)