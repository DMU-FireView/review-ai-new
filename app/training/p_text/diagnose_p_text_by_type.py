"""학습된 KoELECTRA P_text 모델을 진단용 reason type별로 평가하는 모듈.

역할:
- [AI 진단] baseline의 seed 42 split을 재현하고 test review text only로 추론한
  뒤 SUSPICIOUS 예측에 진단 metadata를 사후 결합한다.
- validation에서 선택된 고정 threshold 0.30으로 전체 및 유형별 recall을
  보고한다. 이 모듈에서는 threshold 선택이나 학습을 수행하지 않는다.

수정 범위:
- 운영 runtime이 아닌 오프라인 분석 코드다.
- 전처리, split 정책, 모델 입력 또는 평가 의미 변경은 AI 담당자 검토가
  필요하며 crawler/API 연동은 다른 영역에서 변경해야 한다.

주의:
- diagnostic_type은 새로운 Ground Truth label이 아닌 분석 그룹이다.
- NETWORK_STRONG은 다른 리뷰와 비교해야 강한 근거가 생기는 유형이므로,
  이 유형의 미탐지만으로 P_text 실패라고 판단하지 않는다.
- collection_reason, 재검수근거, diagnostic_type은 모델 입력이 아니다.
- dataset, model 및 진단 CSV는 읽기 전용이며 artifact를 저장하거나 덮어쓰지
  않는다.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import normalize_text, prepare_records, read_review_master, stratified_split


DEFAULT_DATA_PATH = "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
DEFAULT_DIAGNOSIS_PATH = "artifacts/diagnostics/suspicious_type_diagnosis_v2.csv"
DEFAULT_MODEL_PATH = "artifacts/p_text_baseline_v4_3/model"
THRESHOLD = 0.30
DIAGNOSTIC_TYPES = ("TEXT_STRONG", "TEXT_WEAK", "NETWORK_STRONG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--diagnosis-path", default=DEFAULT_DIAGNOSIS_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_diagnoses(path: str) -> dict[str, deque[dict[str, str]]]:
    required = {"master_id", "review", "diagnostic_type"}
    by_text: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Diagnostic CSV is missing columns: {', '.join(missing)}")
        for row in reader:
            diagnostic_type = row["diagnostic_type"].strip()
            if diagnostic_type not in DIAGNOSTIC_TYPES:
                raise ValueError(f"Unexpected diagnostic_type: {diagnostic_type}")
            by_text[normalize_text(row["review"])].append(dict(row))
    return by_text


def attach_diagnoses(
    test_rows: Sequence[Mapping[str, Any]],
    diagnoses: Mapping[str, deque[dict[str, str]]],
) -> list[dict[str, str]]:
    """Join metadata after splitting; metadata never becomes a model input."""
    attached: list[dict[str, str]] = []
    for row in test_rows:
        if int(row["label"]) != 1:
            continue
        key = normalize_text(str(row["text"]))
        candidates = diagnoses.get(key)
        if not candidates:
            raise ValueError(f"No diagnostic CSV match for test review: {row['text']!r}")
        metadata = candidates[0]
        attached.append(
            {
                "master_id": metadata["master_id"],
                "review": str(row["text"]),
                "diagnostic_type": metadata["diagnostic_type"],
            }
        )
    return attached


def predict_probabilities(
    texts: Sequence[str],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> list[float]:
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            # Only review text is tokenized. Diagnostic metadata stays outside
            # the model and is used solely for the grouped report.
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probabilities


def print_result(row: Mapping[str, str], probability: float) -> None:
    detected = probability >= THRESHOLD
    print(f"\nmaster_id: {row['master_id']}")
    print(f"diagnostic_type: {row['diagnostic_type']}")
    print(f"probability: {probability:.6f}")
    print(f"threshold: {THRESHOLD:.2f}")
    print(f"prediction: {'SUSPICIOUS' if detected else 'NORMAL'}")
    print(f"detected: {detected}")
    print(f"review: {row['review']}")


def print_group_summary(rows: Sequence[Mapping[str, str]], detected: Sequence[bool]) -> None:
    total = len(rows)
    true_positives = sum(detected)
    print("\nTOTAL SUSPICIOUS")
    print(f"total: {total}")
    print(f"TP: {true_positives}")
    print(f"FN: {total - true_positives}")
    print(f"Recall: {true_positives / total:.4f}" if total else "Recall: N/A")

    for diagnostic_type in DIAGNOSTIC_TYPES:
        flags = [
            flag
            for row, flag in zip(rows, detected, strict=True)
            if row["diagnostic_type"] == diagnostic_type
        ]
        found = sum(flags)
        print(f"\n{diagnostic_type}")
        print(f"total: {len(flags)}")
        print(f"detected: {found}")
        print(f"missed: {len(flags) - found}")
        print(f"recall: {found / len(flags):.4f}" if flags else "recall: N/A")

    print("\n해석")
    print("TEXT_STRONG: 단일 텍스트에서 비교적 강한 P_text 신호가 나타나는 진단 그룹입니다.")
    print("TEXT_WEAK: 정상 소비자 언어와 겹칠 수 있는 약한 텍스트 신호의 진단 그룹입니다.")
    print("NETWORK_STRONG: 다른 리뷰와 비교해야 강한 근거가 생기는 P_network 성격의 그룹입니다.")
    print("이 그룹의 KoELECTRA 미탐지만으로 P_text 실패라고 판단하지 않습니다.")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise SystemExit("--batch-size and --max-length must be at least 1")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install with: pip install -e '.[ml]'") from exc

    prepared = prepare_records(read_review_master(args.data_path))
    splits = stratified_split(prepared.examples, seed=args.seed)
    rows = attach_diagnoses(splits["test"], load_diagnoses(args.diagnosis_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    probabilities = predict_probabilities(
        [row["review"] for row in rows],
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    detected = [probability >= THRESHOLD for probability in probabilities]
    for row, probability in zip(rows, probabilities, strict=True):
        print_result(row, probability)
    print_group_summary(rows, detected)


if __name__ == "__main__":
    main()
