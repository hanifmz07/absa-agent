from typing import List, Dict


def calculate_instance_metrics(prediction: List[Dict[str, str]], target: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for a single instance.

    Args:
        prediction (List[Dict[str, str]]): Predicted triplets for one sentence.
        target (List[Dict[str, str]]): Target triplets for one sentence.

    Returns:
        Dict[str, float]: A dictionary containing precision, recall, and F1 score.
    """
    true_positive = sum(1 for t in target if t in prediction)
    false_negative = sum(1 for t in target if t not in prediction)
    false_positive = sum(1 for p in prediction if p not in target)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_metrics(predictions: List[List[Dict[str, str]]], targets: List[List[Dict[str, str]]], task='') -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for the given predictions and targets for ABSA.

    Args:
        predictions (List[List[Dict[str, str]]]): List of predicted triplets.
        targets (List[List[Dict[str, str]]]): List of target triplets.
        task (str): The task name for which metrics are calculated.
    
    Returns:
        Dict[str, float]: A dictionary containing precision, recall, and F1 score.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction,target in zip(predictions,targets):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
        false_positive += sum(1 for pred in prediction if pred not in target)
    precision = true_positive/(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * recall * precision)/(recall + precision) if (recall + precision) > 0 else 0
    return {
        f"precision{'_' + task if task else ''}" : precision,
        f"recall{'_' + task if task else ''}" : recall,
        f"f1{'_' + task if task else ''}" : f1
    }


