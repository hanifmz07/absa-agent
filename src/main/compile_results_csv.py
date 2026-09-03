import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

METADATA_FIELDS = [
    "model_name",
    "dataset_type",
    "lang",
    "dataset_folder",
    "prompt_dir",
    "max_retries",
    "seed",
    "run_id",
]

EVAL_METHODS = {
    "exact_match": "exact_match.json",
    "instruct_absa": "instruct_absa.json",
    "semantic": "semantic_metrics.json",
}


def find_run_dirs(results_dir: str) -> List[str]:
    run_dirs = []
    for root, _dirs, files in os.walk(results_dir):
        if "inference_results.json" in files:
            run_dirs.append(root)
    return sorted(run_dirs)


def load_summary(run_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    summary_path = os.path.join(run_dir, filename)
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compile_row(run_dir: str) -> Optional[Dict[str, Any]]:
    summaries = {
        method: load_summary(run_dir, filename)
        for method, filename in EVAL_METHODS.items()
    }

    metadata_source = next((s for s in summaries.values() if s is not None), None)
    if metadata_source is None:
        return None

    row: Dict[str, Any] = {field: metadata_source.get(field) for field in METADATA_FIELDS}
    row["path"] = metadata_source.get("path", run_dir)

    for method, summary in summaries.items():
        for metric in ("precision", "recall", "f1"):
            row[f"{method}_{metric}"] = summary.get(metric) if summary else ""

    return row


def compile_results(results_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for run_dir in find_run_dirs(results_dir):
        row = compile_row(run_dir)
        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda r: tuple(
            str(r.get(field) or "") for field in ("lang", "prompt_dir", "seed", "run_id")
        )
    )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames = METADATA_FIELDS + [
        f"{method}_{metric}"
        for method in EVAL_METHODS
        for metric in ("precision", "recall", "f1")
    ] + ["path"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile per-run eval summary JSONs into a single CSV."
    )
    parser.add_argument(
        "--results_dir",
        default="results/fewshot_single_pass",
        help="Root directory to walk for run directories (default: results/fewshot_single_pass).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <results_dir>/compiled_results.csv).",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.results_dir, "compiled_results.csv")

    rows = compile_results(args.results_dir)
    write_csv(rows, output_path)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
