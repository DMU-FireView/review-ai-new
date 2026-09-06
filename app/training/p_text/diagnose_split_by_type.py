"""SUSPICIOUS 진단 유형의 split별 분포와 P_text 성능을 분석하는 모듈.

역할:
- train/validation/test에 배치된 SUSPICIOUS 진단 유형의 분포를 확인한다.
- 고정 threshold 0.30에서 validation/test의 유형별 탐지 성능을 진단한다.

수정 범위:
- [AI 진단]
- production runtime과 분리된 오프라인 진단 코드다.

주의:
- 모델 재학습을 수행하지 않는다.
- 원본 dataset, CSV 및 model artifact를 수정하지 않는다.
- master_id, collection_reason, 재검수근거, diagnostic_type은 사후 진단용
  metadata이며 모델 입력으로 사용하지 않는다. 모델 입력은 review text only다.
- 진단 결과를 새로운 Ground Truth로 간주하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import LABEL_MAPPING, normalize_text, prepare_records, read_review_master, stratified_split


DEFAULT_DATA_PATH = "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
DEFAULT_DIAGNOSIS_PATH = "artifacts/diagnostics/suspicious_type_diagnosis_v2.csv"
DEFAULT_MODEL_PATH = "artifacts/p_text_baseline_v4_3"
THRESHOLD = 0.30
SEED = 42
EXPECTED_REVIEWED_SUSPICIOUS = 92
DIAGNOSTIC_TYPES = ("TEXT_STRONG", "TEXT_WEAK", "NETWORK_STRONG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--diagnosis-path", default=DEFAULT_DIAGNOSIS_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def canonical_master_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def prepare_master_rows(
    source_rows: Iterable[Mapping[str, Any]],
    official_examples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recover the retained master_id while verifying official preprocessing."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_master_ids: set[str] = set()
    for source in source_rows:
        label_name = str(source.get("final_label", "")).strip().upper()
        text = str(source.get("review", "")).strip()
        if label_name not in LABEL_MAPPING or not text:
            continue
        master_id = canonical_master_id(source.get("master_id"))
        if not master_id:
            raise ValueError("Usable workbook row has an empty master_id")
        if master_id in seen_master_ids:
            raise ValueError(f"Duplicate master_id in usable workbook rows: {master_id}")
        seen_master_ids.add(master_id)
        groups[normalize_text(text)].append(
            {"master_id": master_id, "text": text, "label": LABEL_MAPPING[label_name]}
        )

    retained: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        if len({int(row["label"]) for row in group}) > 1:
            continue
        retained[key] = group[0]

    official_by_text: dict[str, Mapping[str, Any]] = {}
    for example in official_examples:
        key = normalize_text(str(example["text"]))
        if key in official_by_text:
            raise ValueError(
                "Official preprocessing did not produce unique normalized text; "
                "safe split-to-master_id mapping is impossible."
            )
        official_by_text[key] = example

    if set(retained) != set(official_by_text):
        raise ValueError("Internal master_id recovery does not match prepare_records output")
    for key, example in official_by_text.items():
        if retained[key]["text"] != str(example["text"]) or retained[key]["label"] != int(
            example["label"]
        ):
            raise ValueError("Recovered representative differs from prepare_records output")
    return retained


def load_diagnoses(path: str) -> dict[str, dict[str, str]]:
    required = {
        "master_id",
        "review",
        "collection_reason",
        "재검수근거",
        "diagnostic_type",
    }
    diagnoses: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Diagnostic CSV is missing columns: {', '.join(missing)}")
        for source in reader:
            master_id = canonical_master_id(source.get("master_id"))
            if not master_id or master_id in diagnoses:
                raise ValueError(f"Missing or duplicate diagnostic master_id: {master_id!r}")
            diagnostic_type = str(source.get("diagnostic_type", "")).strip()
            if diagnostic_type not in DIAGNOSTIC_TYPES:
                raise ValueError(
                    f"Unexpected diagnostic_type for master_id={master_id}: {diagnostic_type!r}"
                )
            diagnoses[master_id] = {
                key: "" if value is None else str(value).strip()
                for key, value in source.items()
                if key is not None
            }
    if len(diagnoses) != EXPECTED_REVIEWED_SUSPICIOUS:
        raise ValueError(
            f"Expected {EXPECTED_REVIEWED_SUSPICIOUS} reviewed SUSPICIOUS diagnoses, "
            f"found {len(diagnoses)}"
        )
    return diagnoses


def attach_split_metadata(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    retained_by_text: Mapping[str, Mapping[str, Any]],
    diagnoses_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Attach diagnoses by master_id after a uniqueness-checked split lookup."""
    attached = {"train": [], "validation": [], "test": []}
    assigned_ids: set[str] = set()
    for split_name in attached:
        for example in splits[split_name]:
            if int(example["label"]) != LABEL_MAPPING["SUSPICIOUS"]:
                continue
            key = normalize_text(str(example["text"]))
            master_row = retained_by_text.get(key)
            if master_row is None:
                raise ValueError(f"Split review has no verified workbook representative: {key!r}")
            master_id = str(master_row["master_id"])
            if master_id in assigned_ids:
                raise ValueError(f"master_id assigned to multiple splits: {master_id}")
            diagnosis = diagnoses_by_id.get(master_id)
            if diagnosis is None:
                raise ValueError(f"No diagnostic CSV row for retained master_id={master_id}")
            if normalize_text(diagnosis["review"]) != key:
                raise ValueError(
                    f"Workbook/CSV review mismatch for master_id={master_id}; refusing unsafe join"
                )
            attached[split_name].append(
                {
                    "master_id": master_id,
                    "split": split_name,
                    "review": str(example["text"]),
                    "diagnostic_type": diagnosis["diagnostic_type"],
                }
            )
            assigned_ids.add(master_id)

    expected_usable = sum(
        int(row["label"]) == LABEL_MAPPING["SUSPICIOUS"]
        for rows in splits.values()
        for row in rows
    )
    if len(assigned_ids) != expected_usable:
        raise ValueError("Not every usable SUSPICIOUS row received exactly one master_id")
    return attached


def predict_probabilities(
    rows: Sequence[Mapping[str, str]],
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
        for start in range(0, len(rows), batch_size):
            encoded = tokenizer(
                [row["review"] for row in rows[start : start + batch_size]],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probabilities


def print_distribution(
    diagnoses_by_id: Mapping[str, Mapping[str, str]],
    attached: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    reviewed = Counter(row["diagnostic_type"] for row in diagnoses_by_id.values())
    print("SUSPICIOUS TYPE DISTRIBUTION")
    print(f"\n전체 재검수 SUSPICIOUS: {sum(reviewed.values())}")
    print(f"binary usable SUSPICIOUS: {sum(len(rows) for rows in attached.values())}")
    for diagnostic_type in DIAGNOSTIC_TYPES:
        counts = {
            name: sum(row["diagnostic_type"] == diagnostic_type for row in rows)
            for name, rows in attached.items()
        }
        print(f"\n{diagnostic_type}")
        print(f"reviewed_total: {reviewed[diagnostic_type]}")
        print(f"usable_total: {sum(counts.values())}")
        for name in ("train", "validation", "test"):
            print(f"{name}: {counts[name]}")

    print("\nTOTAL")
    print(f"reviewed_total: {sum(reviewed.values())}")
    print(f"usable_total: {sum(len(rows) for rows in attached.values())}")
    for name in ("train", "validation", "test"):
        print(f"{name}: {len(attached[name])}")


def print_performance(
    split_name: str,
    rows: Sequence[Mapping[str, str]],
    probabilities: Sequence[float],
) -> None:
    detected = [probability >= THRESHOLD for probability in probabilities]
    print(f"\n{split_name.upper()} PERFORMANCE (threshold={THRESHOLD:.2f})")
    groups = (("TOTAL", None), *((name, name) for name in DIAGNOSTIC_TYPES))
    for heading, diagnostic_type in groups:
        flags = [
            flag
            for row, flag in zip(rows, detected, strict=True)
            if diagnostic_type is None or row["diagnostic_type"] == diagnostic_type
        ]
        found = sum(flags)
        print(f"\n{heading}")
        print(f"total: {len(flags)}")
        print(f"detected: {found}")
        print(f"missed: {len(flags) - found}")
        print(f"recall: {found / len(flags):.4f}" if flags else "recall: N/A")


def print_text_strong_details(
    attached: Mapping[str, Sequence[Mapping[str, str]]],
    probabilities: Mapping[str, Sequence[float]],
) -> None:
    print("\nTEXT_STRONG DETAILS")
    for split_name in ("train", "validation", "test"):
        print(f"\n{split_name.upper()}")
        rows = attached[split_name]
        probability_by_index = probabilities.get(split_name, ())
        for index, row in enumerate(rows):
            if row["diagnostic_type"] != "TEXT_STRONG":
                continue
            print(f"\nmaster_id: {row['master_id']}")
            print(f"split: {split_name}")
            if split_name != "train":
                probability = probability_by_index[index]
                detected = probability >= THRESHOLD
                print(f"probability: {probability:.6f}")
                print(f"threshold: {THRESHOLD:.2f}")
                print(f"prediction: {'SUSPICIOUS' if detected else 'NORMAL'}")
                print(f"detected: {detected}")
            print(f"review: {row['review']}")


def resolve_model_directory(path: str) -> Path:
    root = Path(path)
    model_directory = root / "model" if (root / "model").is_dir() else root
    if not model_directory.is_dir():
        raise ValueError(f"Model directory not found: {model_directory}")
    return model_directory


def print_interpretation() -> None:
    print("\n해석")
    print("TEXT_STRONG은 단일 리뷰 텍스트에서 강한 P_text 신호가 있다고 진단한 그룹입니다.")
    print("TEXT_WEAK은 정상 소비자 표현과 겹칠 수 있는 약한 신호의 그룹입니다.")
    print("NETWORK_STRONG은 리뷰 간 비교가 핵심인 P_network 성격의 그룹입니다.")
    print("train 결과는 일반화 성능으로 해석하지 않습니다.")
    print("validation은 threshold 및 개발 판단용입니다.")
    print("test는 최종 held-out 진단용이며 test 결과를 보고 threshold를 변경하지 않습니다.")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise SystemExit("--batch-size and --max-length must be at least 1")

    source_rows = read_review_master(args.data_path)
    prepared = prepare_records(source_rows)
    splits = stratified_split(prepared.examples, seed=SEED)
    retained = prepare_master_rows(source_rows, prepared.examples)
    diagnoses = load_diagnoses(args.diagnosis_path)
    attached = attach_split_metadata(splits, retained, diagnoses)
    print_distribution(diagnoses, attached)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install with: pip install -e '.[ml]'") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_directory = resolve_model_directory(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory).to(device)
    model.eval()

    probabilities: dict[str, list[float]] = {}
    for split_name in ("validation", "test"):
        probabilities[split_name] = predict_probabilities(
            attached[split_name],
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print_performance(split_name, attached[split_name], probabilities[split_name])

    print_text_strong_details(attached, probabilities)
    print_interpretation()


if __name__ == "__main__":
    main()
