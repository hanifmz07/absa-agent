"""Evaluate ABSA agent results.

Reads a results JSON file where each entry is shaped as::

    {
        "id": ...,
        "target_text":     [{"aspect": ..., "opinion": ..., "sentiment": ...}, ...],
        "final_extraction":[{"aspect": ..., "opinion": ..., "sentiment": ...}, ...],
        "status": "success" | "failed",
        "attempts": int,
        ...
    }

and computes precision, recall, and F1 via the same ``calculate_metrics``
function used in ``eval.py``.

Key difference from ``eval.py``: the agent results already store structured
dicts (no MVP / GAS string parsing required), so we can feed them directly
into ``calculate_metrics`` after optional normalisation.

Usage::

    # as a module (recommended)
    python -m src.main.eval_agent --results_path results/absa_results.json

    # with options
    python -m src.main.eval_agent \\
        --results_path results/absa_results_async.json \\
        --output_dir   results/eval \\
        --lowercase
"""

import argparse
import json
import logging
import os
from typing import Dict, List
from glob import glob

from ..utils.eval_utils import calculate_metrics, calculate_instance_metrics
from ..utils.parsing import lower_triplets, punctuation_triplets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_triplet(t: Dict) -> bool:
    """Return ``True`` for a well-formed ABSA triplet dict.

    Rejects dicts that contain an ``"error"`` key — these are the error
    sentinels the agent inserts when JSON parsing fails.
    """
    return (
        isinstance(t, dict)
        and "error" not in t
        and {"aspect", "opinion", "sentiment"}.issubset(t.keys())
    )


def _normalise(triplets: List[Dict], lowercase: bool) -> List[Dict]:
    """Apply punctuation-spacing and optional lowercasing normalisation.

    Matches the normalisation pipeline used during SFT evaluation in
    ``eval.py``.

    Args:
        triplets: List of ``{"aspect", "opinion", "sentiment"}`` dicts.
        lowercase: When ``True`` all string values are lowercased.

    Returns:
        Normalised copy of *triplets*.
    """
    result = punctuation_triplets(triplets)
    if lowercase:
        result = lower_triplets(result)
    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(results: List[Dict], lowercase: bool) -> Dict:
    """Compute overall and per-status metrics for a loaded results list.

    Args:
        results: Parsed agent results JSON.
        lowercase: Whether to lowercase triplet values before comparison.

    Returns:
        A dict with keys:

        - ``metrics`` – overall precision / recall / F1 (scaled 0–100).
        - ``n_total``, ``n_success``, ``n_failed``, ``n_skipped`` – counts.
        - ``per_status_metrics`` – same metrics broken down by agent status.
    """
    predictions: List[List[Dict]] = []
    targets: List[List[Dict]] = []
    instance_metrics: List[Dict] = []

    per_status: Dict[str, Dict[str, list]] = {
        "success": {"predictions": [], "targets": []},
        "failed":  {"predictions": [], "targets": []},
    }
    n_skipped = 0

    for entry in results:
        raw_pred = entry.get("final_extraction")
        raw_tgt  = entry.get("target_text")

        if not isinstance(raw_pred, list) or not isinstance(raw_tgt, list):
            logger.warning(
                f"[ID {entry.get('id')}] Skipping – missing or non-list "
                f"'final_extraction' / 'target_text'."
            )
            n_skipped += 1
            continue

        # Drop error-sentinel dicts produced by the agent on parse failures.
        pred = [t for t in raw_pred if _is_valid_triplet(t)]
        tgt  = [t for t in raw_tgt  if _is_valid_triplet(t)]

        if len(pred) != len(raw_pred):
            logger.debug(
                f"[ID {entry.get('id')}] Dropped "
                f"{len(raw_pred) - len(pred)} malformed prediction triplet(s)."
            )

        pred = _normalise(pred, lowercase)
        tgt  = _normalise(tgt,  lowercase)

        predictions.append(pred)
        targets.append(tgt)

        # Per-instance metrics.
        inst_m = calculate_instance_metrics(pred, tgt)
        instance_metrics.append({
            "id":        entry.get("id"),
            "status":    entry.get("status", "unknown"),
            **{k: round(v * 100, 4) for k, v in inst_m.items()},
        })
        logger.debug(
            f"[ID {entry.get('id')}] instance metrics: {inst_m}"
        )

        status = entry.get("status", "unknown")
        if status in per_status:
            per_status[status]["predictions"].append(pred)
            per_status[status]["targets"].append(tgt)

    # Overall metrics, scaled to 0–100 to match eval.py convention.
    logger.debug(f"Predictions: {predictions}")
    logger.debug(f"Targets: {targets}")
    raw = calculate_metrics(predictions, targets, task="")
    metrics = {k: v * 100 for k, v in raw.items()}

    # Per-status metrics.
    per_status_metrics: Dict[str, Dict] = {}
    for status, data in per_status.items():
        if data["predictions"]:
            raw_s = calculate_metrics(
                data["predictions"], data["targets"], task=status
            )
            per_status_metrics[status] = {k: v * 100 for k, v in raw_s.items()}
        else:
            per_status_metrics[status] = {}

    n_success = len(per_status["success"]["predictions"])
    n_failed  = len(per_status["failed"]["predictions"])

    return {
        "metrics": metrics,
        "n_total":   len(predictions),
        "n_success": n_success,
        "n_failed":  n_failed,
        "n_skipped": n_skipped,
        "per_status_metrics": per_status_metrics,
        "instance_metrics": instance_metrics,
    }

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:

    results_path = f"results"
    results_path = os.path.join(results_path, args.runner_type)
    results_path = os.path.join(results_path, args.prompt_set)
    results_path = os.path.join(results_path, f"max_retries_{args.max_retries}")
    results_path = os.path.join(results_path, f"seed_{args.seed}")

    # Check for all folders in the results_path, and find the most recent file
    # If there are no folders, raise an error.
    # results/async/exp1/max_retries_3/seed_42/2026-02-23_19-57-13/inference_results.json -> this is the expected path structure, we want to find the most recent inference_results.json file in the results_path
    if not os.path.exists(results_path):
        logger.error(f"Results path does not exist: {results_path}")
        return

    # Find the most recent inference_results.json under timestamp subdirectories.
    # Expected structure: results/<runner>/<prompt_set>/max_retries_N/seed_S/<timestamp>/inference_results.json
    pattern = os.path.join(results_path, "*", "inference_results.json")
    matching_files = glob(pattern)

    if not matching_files:
        logger.error(f"No inference_results.json found under: {results_path}")
        return

    results_file = max(matching_files, key=os.path.getmtime)
    logger.info(f"Loading results from: {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} entries.")

    report = evaluate(results, lowercase=args.lowercase)

    logger.info(
        f"Total: {report['n_total']}  |  "
        f"Success: {report['n_success']}  |  "
        f"Failed: {report['n_failed']}  |  "
        f"Skipped: {report['n_skipped']}"
    )
    logger.info(f"Overall metrics: {report['metrics']}")
    for status, m in report["per_status_metrics"].items():
        if m:
            logger.info(f"  [{status}] {m}")

    # Log summary.
    logger.info(f"{'='*45}")
    logger.info(f"  Results file : {results_file}")
    logger.info(
        f"  Total        : {report['n_total']}  "
        f"(success={report['n_success']}, failed={report['n_failed']}, "
        f"skipped={report['n_skipped']})"
    )
    logger.info(f"{'='*45}")

    logger.info("=== Overall Metrics ===")
    for k, v in report["metrics"].items():
        logger.info(f"  {k:<25} {v:.2f}")

    for status, m in report["per_status_metrics"].items():
        if m:
            count = report[f"n_{status}"]
            logger.info(f"=== Metrics — status='{status}' ({count} examples) ===")
            for k, v in m.items():
                logger.info(f"  {k:<25} {v:.2f}")
    

    
    output_file = os.path.join(os.path.dirname(results_file), "agent_evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    logger.info(f"Evaluation report saved to {output_file}")


if __name__ == "__main__":
    from logging_config import setup_logging
    os.makedirs("logs", exist_ok=True)
    setup_logging(level=logging.DEBUG, log_file="logs/eval.log")

    parser = argparse.ArgumentParser(
        description="Evaluate ABSA agent results (target_text vs final_extraction)."
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
        "--runner_type",
        type=str,
        default="async",
        choices=["async", "sequential", "batch"],
        help="Type of ABSA agent runner used to generate the results (default: %(default)s).",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase all triplet values before comparison.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
