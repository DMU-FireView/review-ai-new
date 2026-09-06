"""P_text 평가용 이진 분류 지표를 가벼운 의존성으로 계산하는 모듈.

역할: [AI 학습] sklearn 없이 이진 label과 예측값으로 accuracy, class별 지표와
confusion matrix를 계산한다.
수정 범위: metric 정의 변경은 학습 모델 선택과 진단 해석에 영향을 주므로
AI 담당자 검토가 필요하다.
주의: 비어 있지 않은 이진 입력만 허용하며, 운영 scorer나 Ground Truth label을
만드는 기능이 아니라 평가 전용 로직이다.
"""

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
