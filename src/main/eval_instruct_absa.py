import argparse
import glob
import json
import os
from typing import Any, Dict, List

from ..utils.eval_utils import metrics_instructabsa, normalize_triplet_list, to_triplet_string


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


def triplets_to_aoste_text(triplets: List[Dict[str, str]]) -> str:
    return ", ".join(to_triplet_string(triplet) for triplet in normalize_triplet_list(triplets))


def split_aoste_triplets(aoste_text: str) -> List[str]:
    return [part.strip() for part in aoste_text.split(",") if part.strip()]


def instruct_absa_fp_fn(target_aoste: str, prediction_aoste: str) -> Dict[str, List[str]]:
    target_triplets = split_aoste_triplets(target_aoste)
    prediction_triplets = split_aoste_triplets(prediction_aoste)

    false_negative = [
        gt
        for gt in target_triplets
        if not any(pred in gt or gt in pred for pred in prediction_triplets)
    ]
    false_positive = [
        pred
        for pred in prediction_triplets
        if not any(pred in gt or gt in pred for gt in target_triplets)
    ]
    return {
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def evaluate_file(inference_path: str) -> Dict[str, Any]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_aoste = [triplets_to_aoste_text(item.get("target_text", [])) for item in data]
    prediction_aoste = [triplets_to_aoste_text(item.get("final_extraction", [])) for item in data]

    precision, recall, f1, _ = metrics_instructabsa(target_aoste, prediction_aoste)

    result = extract_metadata(inference_path)
    result.update(
        {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
        }
    )
    return result


def evaluate_file_details(inference_path: str) -> List[Dict[str, Any]]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    details: List[Dict[str, Any]] = []
    for item in data:
        target_triplets = normalize_triplet_list(item.get("target_text", []))
        prediction_triplets = normalize_triplet_list(item.get("final_extraction", []))
        target_aoste = triplets_to_aoste_text(target_triplets)
        prediction_aoste = triplets_to_aoste_text(prediction_triplets)

        precision, recall, f1, _ = metrics_instructabsa([target_aoste], [prediction_aoste])
        mismatch = instruct_absa_fp_fn(target_aoste, prediction_aoste)

        details.append(
            {
                "id": item.get("id"),
                "input_text": item.get("input_text"),
                "target_text": target_triplets,
                "final_extraction": prediction_triplets,
                "target_aoste": target_aoste,
                "prediction_aoste": prediction_aoste,
                "false_positive": mismatch["false_positive"],
                "false_negative": mismatch["false_negative"],
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
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
        summary_path = os.path.join(output_dir, "instruct_absa.json")
        detail_path = os.path.join(output_dir, "instruct_absa_detail.json")

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        with open(detail_path, "w", encoding="utf-8") as file:
            json.dump(detail_result, file, indent=2, ensure_ascii=False)

        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved detail metrics to: {detail_path}")

    if len(results) == 1:
        print("InstructABSA metrics:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("InstructABSA metrics summary:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InstructABSA metrics from agent inference results")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path or glob pattern to inference_results.json",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)