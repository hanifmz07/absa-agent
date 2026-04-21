import argparse
import glob
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from ..utils.eval_utils import calculate_metrics_semantic, to_absa_tagged_list


instruction = """Retrieve semantically similar text.
The text that will be retrieved are a tuple of Aspect-Based Sentiment Analysis task containing the aspect term, opinion term, and sentiment with format "[A] aspect term [O] opinion term [S] sentiment".
The aspect and opinion can be a subset of the other text as long as it is not contradictive.
The aspect can be null if there is no aspect term (implicit aspect), but the opinion term must exist.
The sentiment should be the same for both texts.
"""


def format_sts(text: str) -> str:
    return f"Instruct: {instruction}\\nQuery: {text}"


def semantic_fp_fn(
    prediction_list: List[str],
    target_list: List[str],
    model: SentenceTransformer,
    threshold: float = 0.9,
) -> Dict[str, List[str]]:
    false_negative_candidates = [target for target in target_list if target not in prediction_list]
    false_positive_candidates = [pred for pred in prediction_list if pred not in target_list]

    remaining_false_positive = list(false_positive_candidates)
    remaining_false_negative: List[str] = []

    for false_negative_candidate in false_negative_candidates:
        emb_false_negative = model.encode(format_sts(str(false_negative_candidate)))
        matched_index = None

        for idx, false_positive_candidate in enumerate(remaining_false_positive):
            emb_false_positive = model.encode(format_sts(str(false_positive_candidate)))
            score = util.cos_sim(emb_false_negative, emb_false_positive)
            if score.item() >= threshold:
                matched_index = idx
                break

        if matched_index is not None:
            remaining_false_positive.pop(matched_index)
        else:
            remaining_false_negative.append(false_negative_candidate)

    return {
        "false_positive": remaining_false_positive,
        "false_negative": remaining_false_negative,
    }


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


def evaluate_file(inference_path: str, model: SentenceTransformer) -> Dict[str, Any]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_lists = [to_absa_tagged_list(item.get("target_text", [])) for item in data]
    prediction_lists = [to_absa_tagged_list(item.get("final_extraction", [])) for item in data]
    scores = calculate_metrics_semantic(prediction_lists, target_lists, model, task="semantic")

    result = extract_metadata(inference_path)
    result.update(
        {
            "precision": scores["precision_semantic"] * 100,
            "recall": scores["recall_semantic"] * 100,
            "f1": scores["f1_semantic"] * 100,
        }
    )
    return result


def evaluate_file_details(inference_path: str, model: SentenceTransformer) -> List[Dict[str, Any]]:
    with open(inference_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    details: List[Dict[str, Any]] = []
    for item in data:
        target_list = to_absa_tagged_list(item.get("target_text", []))
        prediction_list = to_absa_tagged_list(item.get("final_extraction", []))
        score = calculate_metrics_semantic([prediction_list], [target_list], model, task="semantic")
        mismatch = semantic_fp_fn(prediction_list, target_list, model)

        details.append(
            {
                "id": item.get("id"),
                "input_text": item.get("input_text"),
                "target_list": target_list,
                "prediction_list": prediction_list,
                "false_positive": mismatch["false_positive"],
                "false_negative": mismatch["false_negative"],
                "precision": score["precision_semantic"] * 100,
                "recall": score["recall_semantic"] * 100,
                "f1": score["f1_semantic"] * 100,
            }
        )

    return details


def main(args: argparse.Namespace) -> None:
    load_dotenv()

    inference_paths = resolve_inference_paths(args.inference_path)
    if not inference_paths:
        raise FileNotFoundError(f"No inference files found from: {args.inference_path}")

    cache_dir = os.getenv("CACHE_DIR")
    model = SentenceTransformer(
        args.embedding_model_name,
        trust_remote_code=True,
        cache_folder=cache_dir,
    )

    results = []
    for inference_path in inference_paths:
        print(f"Processing: {inference_path}")
        summary_result = evaluate_file(inference_path, model)
        detail_result = evaluate_file_details(inference_path, model)
        results.append(summary_result)

        output_dir = os.path.dirname(inference_path)
        summary_path = os.path.join(output_dir, "semantic_metrics.json")
        detail_path = os.path.join(output_dir, "semantic_metrics_detail.json")

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary_result, file, indent=2, ensure_ascii=False)
        with open(detail_path, "w", encoding="utf-8") as file:
            json.dump(detail_result, file, indent=2, ensure_ascii=False)

        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved detail metrics to: {detail_path}")

    if len(results) == 1:
        print("Semantic metrics:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("Semantic metrics summary:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic ABSA metrics from agent inference results")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path or glob pattern to inference_results.json",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        required=True,
        help="Embedding model name, e.g. Qwen/Qwen3-Embedding-8B",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)