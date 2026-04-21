import re
from typing import Dict, List

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover - optional dependency for semantic eval only
    SentenceTransformer = None  # type: ignore[assignment]
    util = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

instruction = """Retrieve semantically similar text.
The text that will be retrieved are a tuple of Aspect-Based Sentiment Analysis task containing the aspect term, opinion term, and sentiment with format "[A] aspect term [O] opinion term [S] sentiment".
The aspect and opinion can be a subset of the other text as long as it is not contradictive.
The aspect can be null if there is no aspect term (implicit aspect), but the opinion term must exist.
The sentiment should be the same for both texts.
"""


def normalize_triplet(triplet: Dict[str, str]) -> Dict[str, str]:
    return {
        "A": str(triplet.get("aspect", triplet.get("A", ""))).strip(),
        "O": str(triplet.get("opinion", triplet.get("O", ""))).strip(),
        "S": str(triplet.get("sentiment", triplet.get("S", ""))).strip(),
    }


def normalize_triplet_list(triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not isinstance(triplets, list):
        return []
    return [normalize_triplet(triplet) for triplet in triplets if isinstance(triplet, dict)]


def to_triplet_string(triplet: Dict[str, str]) -> str:
    normalized = normalize_triplet(triplet)
    return f"{normalized['A']}:{normalized['O']}:{normalized['S']}"


def to_absa_tagged_string(triplet: Dict[str, str]) -> str:
    normalized = normalize_triplet(triplet)
    return f"[A] {normalized['A']} [O] {normalized['O']} [S] {normalized['S']}"


def to_absa_tagged_list(triplets: List[Dict[str, str]]) -> List[str]:
    return [to_absa_tagged_string(triplet) for triplet in normalize_triplet_list(triplets)]


def format_sts(text: str, instruction_text: str) -> str:
    return f"Instruct: {instruction_text}\nQuery: {text}"


def calculate_metrics(
    predictions: List[List[Dict[str, str]]],
    targets: List[List[Dict[str, str]]],
    task: str = "",
) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction, target in zip(predictions, targets):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
        false_positive += sum(1 for pred in prediction if pred not in target)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    return {
        f"precision_{task}": precision,
        f"recall_{task}": recall,
        f"f1_{task}": f1,
    }


def calculate_metrics_semantic(
    predictions: List[List[Dict[str, str]]],
    targets: List[List[Dict[str, str]]],
    model: SentenceTransformer,
    task: str = "",
    threshold: float = 0.9,
) -> Dict[str, float]:
    global instruction

    if util is None:
        raise ImportError(
            "sentence-transformers is required for semantic evaluation. Install it first."
        )

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for prediction, target in tqdm(
        zip(predictions, targets), total=len(predictions), desc="Calculating semantic metrics"
    ):
        false_negative_candidates = []
        false_positive_candidates = []
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative_candidates.append(target_tuple)
        false_positive_candidates += [pred for pred in prediction if pred not in target]

        for false_negative_candidate in false_negative_candidates:
            emb_false_negative = model.encode(format_sts(str(false_negative_candidate), instruction))
            for false_positive_candidate in false_positive_candidates:
                emb_false_positive = model.encode(format_sts(str(false_positive_candidate), instruction))
                score = util.cos_sim(emb_false_negative, emb_false_positive)
                if score.item() >= threshold:
                    true_positive += 1
                    false_positive_candidates.remove(false_positive_candidate)
                    break
            else:
                false_negative += 1
        false_positive += len(false_positive_candidates)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    return {
        f"precision_{task}": precision,
        f"recall_{task}": recall,
        f"f1_{task}": f1,
    }


def parse_absa_string(text: str) -> List[Dict[str, str]]:
    pattern = r"\[(\w+)\]\s*([^[]+)"
    matches = re.findall(pattern, text)

    result: List[Dict[str, str]] = []
    current_dict: Dict[str, str] = {}

    for tag, content in matches:
        if tag == "SSEP":
            result.append(current_dict)
            current_dict = {}
        else:
            current_dict[tag] = content.strip()

    if current_dict:
        result.append(current_dict)

    return result


def parse_aoste(text: str) -> List[Dict[str, str]]:
    pattern = r"([^:]+):([^,]+):(\w+)"
    matches = re.findall(pattern, text)

    result = []
    for aspect, opinion, sentiment in matches:
        result.append({
            "A": aspect.strip(),
            "O": opinion.strip(),
            "S": sentiment.strip(),
        })

    return result


def metrics_instructabsa(y_true, y_pred, is_triplet_extraction=False):
    total_pred = 0
    total_gt = 0
    tp = 0
    if not is_triplet_extraction:
        for gt, pred in zip(y_true, y_pred):
            gt_list = [x.strip() for x in gt.split(",") if x.strip()]
            pred_list = [x.strip() for x in pred.split(",") if x.strip()]
            total_pred += len(pred_list)
            total_gt += len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if pred_val in gt_val or gt_val in pred_val:
                        tp += 1
                        break

    else:
        for gt, pred in zip(y_true, y_pred):
            gt_list = [x.strip() for x in gt.split(",") if x.strip()]
            pred_list = [x.strip() for x in pred.split(",") if x.strip()]
            total_pred += len(pred_list)
            total_gt += len(gt_list)
            for gt_val in gt_list:
                gt_asp = gt_val.split(":")[0].strip()

                try:
                    gt_op = gt_val.split(":")[1].strip()
                except Exception:
                    continue

                try:
                    gt_sent = gt_val.split(":")[2].strip()
                except Exception:
                    continue

                for pred_val in pred_list:
                    pr_asp = pred_val.split(":")[0].strip()

                    try:
                        pr_op = pred_val.split(":")[1].strip()
                    except Exception:
                        continue

                    try:
                        pr_sent = gt_val.split(":")[2].strip()
                    except Exception:
                        continue

                    if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                        tp += 1

    p = tp / total_pred if total_pred > 0 else 0.0
    r = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1, None


