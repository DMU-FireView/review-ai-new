"""v4.3 SUSPICIOUS 리뷰를 진단용 reason type으로 분류하는 모듈.

역할: [AI 진단] 검수 metadata를 읽고 보수적인 규칙에 따라 TEXT_STRONG,
TEXT_WEAK 또는 NETWORK_STRONG 분석 그룹을 부여한다.
수정 범위: reason 정의와 heuristic pattern 변경은 AI 담당자 검토가 필요하며,
이 유형은 모델 입력이나 운영 scoring 정책으로 사용하지 않는다.
주의: 진단 유형은 Ground Truth가 아니고 원본 workbook은 읽기 전용이며, 기존
진단 artifact를 보존하기 위해 CSV를 덮어쓰지 않고 새로 생성한다.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


DATA_PATH = "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
SHEET_NAME = "전체_SUSPICIOUS_재검수"
OUTPUT_PATH = "artifacts/diagnostics/suspicious_type_diagnosis_v2.csv"
EXPECTED_SUSPICIOUS_COUNT = 92
OUTPUT_COLUMNS = (
    "master_id",
    "platform",
    "review",
    "collection_reason",
    "재검수근거",
    "diagnostic_type",
)
TYPE_ORDER = ("TEXT_STRONG", "TEXT_WEAK", "NETWORK_STRONG")

# NETWORK_STRONG is checked first. These expressions explicitly require a
# relationship to other reviews, copies, registrations, or a similarity score.
NETWORK_PATTERNS = (
    r"jaccard|자카드|유사도",
    r"(?:리뷰|문구|내용).{0,12}(?:간|끼리).{0,12}(?:유사|반복|동일)",
    r"(?:동일|유사).{0,10}(?:리뷰|문구|내용).{0,10}(?:여러|다수|등록|작성|존재)",
    r"(?:중복|반복|재).{0,4}등록",
    r"복제|복붙|원본군",
    r"(?:다른|타|여러|다수의?|복수).{0,6}리뷰",
)

PRODUCT_COPY_PATTERNS = (
    r"상세\s*페이지|상품\s*설명|제품\s*설명|공식\s*상품|광고\s*카피|홍보문",
    r"성분.{0,12}(?:나열|중심|설명)|효능.{0,12}(?:나열|중심|설명)|기능.{0,12}(?:나열|중심|설명)",
    r"(?:객관적|설명문).{0,10}(?:문체|형태)|(?:구조|문체).{0,12}(?:광고|홍보|상세페이지)",
)

LACK_OF_EXPERIENCE_PATTERNS = (
    r"(?:개인|실제|직접|사용)\s*(?:사용\s*)?경험.{0,8}(?:없|부족|미약|거의)",
    r"후기성.{0,8}(?:없|부족|미약|거의)|개인.{0,8}(?:맥락|후기).{0,8}(?:없|부족|거의)",
)

PROMOTION_PATTERNS = (
    r"판매\s*촉진|구매\s*유도|판매\s*유도",
    r"반드시\s*(?:사|구매)|무조건\s*(?:사|구매)|꼭\s*(?:사세요|구매)",
    r"적극\s*추천|강력\s*추천",
)

# Concrete situations make promotional wording weak evidence unless the reason
# explicitly says that personal experience is absent.
PERSONAL_EXPERIENCE_PATTERNS = (
    r"설치|반품|교환|사이즈|크기|배송|도착|포장",
    r"직접|사용(?:해|하|중|감)|써\s*보|먹어\s*보|입어\s*보|발라\s*보",
    r"집|거실|방|주방|욕실|회사|차량|아이|가족|부모|선물",
    r"판매자|기사님|색상|재구매|구매했|주문했|받았|며칠|개월|일주일",
)

REGRESSION_EXPECTATIONS = {
    "480": "TEXT_STRONG",
    "472": "TEXT_STRONG",
    "1008": "TEXT_STRONG",
    "1009": "TEXT_STRONG",
    "1010": "TEXT_STRONG",
    "49": "TEXT_WEAK",
    "115": "TEXT_WEAK",
    "141": "TEXT_WEAK",
    "326": "NETWORK_STRONG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    return parser.parse_args()


def normalized(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def has_abnormal_internal_repetition(review: str) -> bool:
    """Detect repeated material within one review, without cross-review data."""
    text = normalized(review)
    if not text:
        return False

    sentences = [
        normalized(part)
        for part in re.split(r"[.!?。！？]+|\n+", text)
        if normalized(part)
    ]
    sentence_counts = Counter(sentences)
    if any(count >= 3 and len(sentence) >= 2 for sentence, count in sentence_counts.items()):
        return True

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.casefold())
    # Three repeats suffice for any non-trivial token/block. Two repeats count
    # only for a longer phrase covering most of the review.
    for block_size in range(1, len(tokens) // 2 + 1):
        for start in range(0, len(tokens) - block_size * 2 + 1):
            block = tokens[start : start + block_size]
            repetitions = 1
            cursor = start + block_size
            while tokens[cursor : cursor + block_size] == block:
                repetitions += 1
                cursor += block_size
            if repetitions >= 3 and len("".join(block)) >= 2:
                return True
            covered = repetitions * block_size
            if repetitions >= 2 and block_size >= 4 and covered / len(tokens) >= 0.8:
                return True
    return False


def classify_reason(row: Mapping[str, Any]) -> str:
    """Apply cross-review, internal-repeat, strong-text, then weak priority."""
    evidence = " ".join(
        filter(
            None,
            (normalized(row.get("재검수근거")), normalized(row.get("collection_reason"))),
        )
    )
    review = normalized(row.get("review"))
    if matches_any(evidence, NETWORK_PATTERNS):
        return "NETWORK_STRONG"
    if has_abnormal_internal_repetition(review):
        return "TEXT_STRONG"
    lacks_experience = matches_any(evidence, LACK_OF_EXPERIENCE_PATTERNS)
    has_personal_experience = matches_any(
        f"{review} {evidence}", PERSONAL_EXPERIENCE_PATTERNS
    )
    if matches_any(evidence, PRODUCT_COPY_PATTERNS) and (
        lacks_experience or not has_personal_experience
    ):
        return "TEXT_STRONG"
    if matches_any(evidence, PROMOTION_PATTERNS) and lacks_experience:
        return "TEXT_STRONG"
    return "TEXT_WEAK"


def canonical_master_id(value: str) -> str:
    return value[:-2] if value.endswith(".0") and value[:-2].isdigit() else value


def validate_regressions(rows: list[dict[str, str]]) -> None:
    """Fail clearly if known examples regress; IDs do not drive classification."""
    actual = {
        canonical_master_id(row["master_id"]): row["diagnostic_type"] for row in rows
    }
    failures = []
    for master_id, expected in REGRESSION_EXPECTATIONS.items():
        observed = actual.get(master_id)
        if observed != expected:
            failures.append(
                f"master_id={master_id}: expected {expected}, got {observed or 'MISSING'}"
            )
    if failures:
        raise SystemExit("Regression validation failed:\n  " + "\n  ".join(failures))


def load_suspicious_rows(path: str) -> list[dict[str, str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise SystemExit("openpyxl is required. Install with: pip install openpyxl") from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if SHEET_NAME not in workbook.sheetnames:
            raise ValueError(f"Workbook does not contain sheet: {SHEET_NAME}")
        values = workbook[SHEET_NAME].iter_rows(values_only=True)
        try:
            headers = [normalized(value) for value in next(values)]
        except StopIteration as exc:
            raise ValueError(f"Sheet is empty: {SHEET_NAME}") from exc

        required = set(OUTPUT_COLUMNS[:-1]) | {"final_label"}
        missing = sorted(required - set(headers))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        results: list[dict[str, str]] = []
        for values_row in values:
            source = dict(zip(headers, values_row, strict=True))
            if normalized(source.get("final_label")).upper() != "SUSPICIOUS":
                continue
            result = {column: normalized(source.get(column)) for column in OUTPUT_COLUMNS[:-1]}
            result["diagnostic_type"] = classify_reason(source)
            results.append(result)
        return results
    finally:
        workbook.close()


def write_csv(rows: list[dict[str, str]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation protects an existing diagnostic result from overwrite.
        with output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing output: {output}") from exc


def shortened(text: str, limit: int = 180) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def print_rows(rows: list[dict[str, str]]) -> None:
    print("\nAll SUSPICIOUS diagnoses")
    for index, row in enumerate(rows, start=1):
        print(f"\n[{index}/{len(rows)}]")
        for column in OUTPUT_COLUMNS:
            value = shortened(row[column]) if column == "review" else row[column]
            print(f"  {column}: {value}")


def print_summary(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["diagnostic_type"] for row in rows)
    total = len(rows)
    print("\nSummary")
    for diagnostic_type in TYPE_ORDER:
        count = counts[diagnostic_type]
        ratio = count / total * 100 if total else 0.0
        print(f"  {diagnostic_type}: {count} ({ratio:.1f}%)")
    print(f"  TOTAL: {total}")

    print("\nRepresentative samples")
    for diagnostic_type in TYPE_ORDER:
        print(f"\n  {diagnostic_type}")
        samples = [row for row in rows if row["diagnostic_type"] == diagnostic_type][:5]
        if not samples:
            print("    (none)")
        for row in samples:
            print(
                f"    master_id={row['master_id']} platform={row['platform']} "
                f"review={shortened(row['review'])!r}"
            )


def main() -> None:
    args = parse_args()
    rows = load_suspicious_rows(args.data_path)
    if len(rows) != EXPECTED_SUSPICIOUS_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_SUSPICIOUS_COUNT} SUSPICIOUS rows, found {len(rows)}; "
            "no output was written."
        )

    validate_regressions(rows)
    print_rows(rows)
    print_summary(rows)
    write_csv(rows, args.output_path)
    print(f"\nCSV created: {args.output_path}")


if __name__ == "__main__":
    main()
