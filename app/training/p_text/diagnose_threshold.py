"""학습된 KoELECTRA P_text baseline의 threshold를 진단하는 모듈.

역할: [AI 진단] baseline split을 재현하고 review text only로 P(SUSPICIOUS)를
계산하여 validation에서 threshold를 선택한 뒤 고정된 test 결과를 보고한다.
수정 범위: threshold 선택 정책이나 전처리 변경은 AI 담당자 검토가 필요하며
운영 연동 목적으로 변경해서는 안 된다.
주의: Ground Truth나 운영 scoring이 아닌 오프라인 분석이며, 모델을 재학습하거나
dataset을 수정하거나 기존 artifact를 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .data import prepare_records, read_review_master, split_counts, stratified_split
from .metrics import classification_metrics


DEFAULT_DATA_PATH = "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
DEFAULT_MODEL_PATH = "artifacts/p_text_baseline_v4_3/model"
LABEL_NAMES = {0: "NORMAL", 1: "SUSPICIOUS"}
THRESHOLDS = tuple(value / 100 for value in range(5, 100, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_probabilities(
    rows: Sequence[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Return P(SUSPICIOUS) in input order without computing gradients."""
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probabilities


def metrics_at_threshold(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    return classification_metrics(labels, predictions)


def select_threshold(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Maximize SUSPICIOUS F1, then recall, then closeness to 0.5."""
    candidates = [
        (threshold, metrics_at_threshold(labels, probabilities, threshold))
        for threshold in THRESHOLDS
    ]
    return max(
        candidates,
        key=lambda item: (
            item[1]["suspicious"]["f1"],
            item[1]["suspicious"]["recall"],
            -abs(item[0] - 0.5),
        ),
    )


def print_metrics(title: str, threshold: float, metrics: dict[str, Any], *, accuracy: bool) -> None:
    suspicious = metrics["suspicious"]
    print(f"\n{title}")
    print(f"  threshold: {threshold:.2f}")
    if accuracy:
        print(f"  accuracy: {metrics['accuracy']:.6f}")
    print(f"  SUSPICIOUS precision: {suspicious['precision']:.6f}")
    print(f"  SUSPICIOUS recall: {suspicious['recall']:.6f}")
    print(f"  SUSPICIOUS F1: {suspicious['f1']:.6f}")
    print(f"  confusion matrix: {metrics['confusion_matrix']}")


def abbreviated(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def print_suspicious_examples(
    dataset_name: str,
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
) -> None:
    print(f"\nActual SUSPICIOUS examples: {dataset_name}")
    for row, probability in zip(rows, probabilities, strict=True):
        if row["label"] != 1:
            continue
        predicted = int(probability >= threshold)
        print(
            f"  [{dataset_name}] actual=SUSPICIOUS "
            f"P(SUSPICIOUS)={probability:.6f} predicted={LABEL_NAMES[predicted]} "
            f"text={abbreviated(row['text'])!r}"
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.max_length < 1:
        raise SystemExit("--max-length must be at least 1")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install with: pip install -e '.[ml]'") from exc

    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")

    prepared = prepare_records(read_review_master(args.data_path))
    splits = stratified_split(prepared.examples, seed=args.seed)
    print(f"Split counts: {split_counts(splits)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"Model: {model_path} (device={device})")

    probabilities: dict[str, list[float]] = {}
    for name in ("validation", "test"):
        probabilities[name] = predict_probabilities(
            splits[name],
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    validation_labels = [row["label"] for row in splits["validation"]]
    print("\nValidation threshold comparison")
    for threshold in THRESHOLDS:
        metrics = metrics_at_threshold(validation_labels, probabilities["validation"], threshold)
        print_metrics("Validation", threshold, metrics, accuracy=False)

    selected_threshold, selected_validation_metrics = select_threshold(
        validation_labels, probabilities["validation"]
    )
    print_metrics(
        "Selected validation threshold",
        selected_threshold,
        selected_validation_metrics,
        accuracy=False,
    )

    test_labels = [row["label"] for row in splits["test"]]
    test_metrics = metrics_at_threshold(
        test_labels, probabilities["test"], selected_threshold
    )
    print_metrics("Final test evaluation", selected_threshold, test_metrics, accuracy=True)

    for name in ("validation", "test"):
        print_suspicious_examples(
            name, splits[name], probabilities[name], selected_threshold
        )


if __name__ == "__main__":
    main()
