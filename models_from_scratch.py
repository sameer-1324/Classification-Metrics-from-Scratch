from __future__ import annotations

import numpy as np


def _validate_classification_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert classification metric inputs.

    This implementation is for binary classification only.

    Expected labels:
    0 = negative class
    1 = positive class
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        raise ValueError("y_true cannot be empty.")

    if y_pred.size == 0:
        raise ValueError("y_pred cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}."
        )

    valid_true_labels = np.isin(y_true, [0, 1])
    valid_pred_labels = np.isin(y_pred, [0, 1])

    if not np.all(valid_true_labels):
        raise ValueError("y_true must contain only binary labels: 0 and 1.")

    if not np.all(valid_pred_labels):
        raise ValueError("y_pred must contain only binary labels: 0 and 1.")

    return y_true.astype(int), y_pred.astype(int)


def confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """
    Compute binary confusion matrix components.

    Returns
    -------
    dict[str, int]
        Dictionary containing TP, TN, FP, FN.

    Positive class = 1
    Negative class = 0
    """

    y_true, y_pred = _validate_classification_inputs(y_true, y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return {
        "TP": int(true_positive),
        "TN": int(true_negative),
        "FP": int(false_positive),
        "FN": int(false_negative),
    }


def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate accuracy.

    Accuracy measures the proportion of all predictions that are correct.

    Formula:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """

    cm = confusion_matrix_binary(y_true, y_pred)

    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    correct = cm["TP"] + cm["TN"]

    return float(correct / total)


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate precision.

    Precision answers:
    Of all predicted positives, how many were actually positive?

    Formula:
    Precision = TP / (TP + FP)
    """

    cm = confusion_matrix_binary(y_true, y_pred)

    denominator = cm["TP"] + cm["FP"]

    if denominator == 0:
        return 0.0

    return float(cm["TP"] / denominator)


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate recall.

    Recall answers:
    Of all actual positives, how many did the model catch?

    Formula:
    Recall = TP / (TP + FN)
    """

    cm = confusion_matrix_binary(y_true, y_pred)

    denominator = cm["TP"] + cm["FN"]

    if denominator == 0:
        return 0.0

    return float(cm["TP"] / denominator)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate F1 score.

    F1 is the harmonic mean of precision and recall.

    Formula:
    F1 = 2 * precision * recall / (precision + recall)
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return float(2 * precision * recall / denominator)