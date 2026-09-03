"""Dependency-light binary classification metrics."""

from __future__ import annotations

from typing import Sequence


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if any(value not in (0, 1) for value in (*y_true, *y_pred)):
        raise ValueError("Only binary labels 0 and 1 are supported")
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred, strict=True))
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred, strict=True))

    def scores(label_tp: int, label_fp: int, label_fn: int) -> dict[str, float]:
        precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
        recall = label_tp / (label_tp + label_fn) if label_tp + label_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    normal = scores(tn, fn, fp)
    suspicious = scores(tp, fp, fn)
    return {
        "accuracy": (tn + tp) / len(y_true),
        "precision": (normal["precision"] + suspicious["precision"]) / 2,
        "recall": (normal["recall"] + suspicious["recall"]) / 2,
        "f1": (normal["f1"] + suspicious["f1"]) / 2,
        "suspicious": suspicious,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
