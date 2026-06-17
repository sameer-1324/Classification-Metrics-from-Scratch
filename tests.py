import numpy as np
import pytest

from models_from_scratch import (
    confusion_matrix_binary,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def test_confusion_matrix_binary() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    result = confusion_matrix_binary(y_true, y_pred)

    assert result["TP"] == 2
    assert result["TN"] == 2
    assert result["FP"] == 1
    assert result["FN"] == 1


def test_accuracy_score() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    result = accuracy_score(y_true, y_pred)

    assert result == pytest.approx(4 / 6)


def test_precision_score() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    result = precision_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_recall_score() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    result = recall_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_f1_score() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    result = f1_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_precision_returns_zero_when_no_positive_predictions() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    result = precision_score(y_true, y_pred)

    assert result == 0.0


def test_recall_returns_zero_when_no_actual_positives() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    result = recall_score(y_true, y_pred)

    assert result == 0.0


def test_f1_returns_zero_when_precision_and_recall_are_zero() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    result = f1_score(y_true, y_pred)

    assert result == 0.0


def test_classification_metrics_raise_error_for_shape_mismatch() -> None:
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        accuracy_score(y_true, y_pred)


def test_classification_metrics_raise_error_for_non_binary_labels() -> None:
    y_true = np.array([1, 0, 2])
    y_pred = np.array([1, 0, 1])

    with pytest.raises(ValueError, match="binary labels"):
        accuracy_score(y_true, y_pred)