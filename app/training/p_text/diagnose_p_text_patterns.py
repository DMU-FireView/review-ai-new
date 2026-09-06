"""TEXT_STRONG 리뷰를 P_text v2 학습 관점의 세부 패턴으로 진단하는 모듈.

역할:
- binary usable 데이터에 남은 TEXT_STRONG 리뷰를 세부 텍스트 패턴으로
  분류하고 P_text v2 positive 학습 후보 여부를 진단한다.

수정 범위:
- [AI 진단]
- production/runtime과 분리된 연구용 진단 코드다.

주의:
- 이 결과는 Ground Truth가 아니며 기존 dataset label을 변경하지 않는다.
- p_text_v2_positive_candidate는 학습 후보 여부일 뿐 정상/조작 확정 의미가 아니다.
- collection_reason과 재검수근거는 보조 참고 정보이며 모델 입력이나 정답으로
  사용하지 않는다. 패턴 분류는 review text 자체를 우선한다.
- 원본 dataset, 진단 CSV 및 model artifact를 수정하거나 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import warnings

from .data import LABEL_MAPPING, normalize_text, prepare_records, read_review_master
from .diagnose_split_by_type import load_diagnoses, prepare_master_rows


DEFAULT_DATA_PATH = "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
DEFAULT_DIAGNOSIS_PATH = "artifacts/diagnostics/suspicious_type_diagnosis_v2.csv"
DEFAULT_OUTPUT_PATH = "artifacts/diagnostics/p_text_pattern_diagnosis_v4.csv"
EXPECTED_REVIEWED_TEXT_STRONG = 28
EXPECTED_USABLE_TEXT_STRONG = 23
PATTERN_ORDER = (
    "REPETITION",
    "PRODUCT_COPY",
    "STRUCTURED_INFO",
    "PROMOTIONAL_CTA",
    "AMBIGUOUS_TEXT_STRONG",
)
POSITIVE_PATTERNS = frozenset(PATTERN_ORDER[:-1])
OUTPUT_COLUMNS = (
    "master_id",
    "review",
    "previous_diagnostic_type",
    "p_text_pattern",
    "p_text_v2_positive_candidate",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--diagnosis-path", default=DEFAULT_DIAGNOSIS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def has_clear_internal_repetition(review: str) -> bool:
    """Require at least three consecutive occurrences of a phrase or sentence."""
    text = " ".join(str(review).split())
    if not text:
        return False

    sentences = [
        " ".join(part.split()).casefold()
        for part in re.split(r"[.!?。！？]+|\n+", str(review))
        if part.strip()
    ]
    if any(count >= 3 and len(sentence) >= 2 for sentence, count in Counter(sentences).items()):
        return True

    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text.casefold())
    for block_size in range(1, len(tokens) // 3 + 1):
        for start in range(0, len(tokens) - block_size * 3 + 1):
            block = tokens[start : start + block_size]
            if len("".join(block)) < 2:
                continue
            if (
                tokens[start + block_size : start + block_size * 2] == block
                and tokens[start + block_size * 2 : start + block_size * 3] == block
            ):
                return True
    return False


def has_structured_information(review: str) -> bool:
    raw_lines = [line.strip() for line in str(review).splitlines()]
    lines = [line for line in raw_lines if line]
    list_markers = sum(
        bool(re.match(r"(?:[-*•▪]|[0-9]+[.)]|✅|✔|☑|📌|💡|▶)", line))
        for line in lines
    )
    labeled_sections = len(
        re.findall(
            r"(?:핵심\s*성분|성분|효능|특징|장점|사용법|추천\s*대상|활용법)\s*[:：]",
            str(review),
            flags=re.IGNORECASE,
        )
    )
    if list_markers >= 2 or labeled_sections >= 3:
        return True

    heading_pattern = re.compile(
        r"^(?:핵심\s*성분|주요\s*성분|효능|특징|사용법)\s*[:：]?\s*$",
        flags=re.IGNORECASE,
    )

    def is_heading(line: str) -> bool:
        without_decoration = re.sub(r"^[^가-힣A-Za-z0-9]+", "", line).strip()
        return bool(heading_pattern.match(without_decoration))

    sentence_ending = re.compile(r"(?:요|습니다|입니다|했어요|합니다)[.!?。！？]?$", re.IGNORECASE)
    for index, line in enumerate(raw_lines):
        if not is_heading(line):
            continue
        item_count = 0
        started = False
        for candidate in raw_lines[index + 1 :]:
            if not candidate:
                if started:
                    break
                continue
            if is_heading(candidate):
                break
            words = re.findall(r"[가-힣A-Za-z0-9%]+", candidate)
            if not (1 <= len(words) <= 6 and 2 <= len(candidate) <= 40):
                break
            if sentence_ending.search(candidate):
                break
            started = True
            item_count += 1
            if item_count >= 3:
                return True
    return False


def has_personal_experience(review: str) -> bool:
    patterns = (
        r"(?:직접|실제로)\s*(?:사용|구매|설치|먹|입|발라|써)",
        r"(?:사진|화면)(?:으로)?\s*(?:봤을?|보았을?|볼)\s*때.{0,50}(?:색|색상|향|예쁘|이쁘|부담|느낌)",
        r"직접\s*(?:보니|보니까|봤을?|착용해\s*보니|사용해\s*보니)",
        r"사용(?:해|했|하면서|중이|후에)|써\s*보|써봤|먹어\s*보|먹었|입어\s*보|입었",
        r"발라\s*보|발랐|설치했|배송받|도착했|주문했|구매했|재구매했",
        r"(?:색|색상|향|착용감|사용감).{0,30}(?:예쁘|이쁘|부담|마음에|느껴|좋았|별로)",
        r"(?:며칠|몇\s*주|일주일|개월|한\s*달|두\s*달).{0,12}(?:사용|복용|써|먹)",
        r"(?:저는|제가|우리\s*집|아이|가족|부모님|남편|아내).{0,18}(?:사용|먹|입|좋|맞)",
    )
    return any(re.search(pattern, str(review), flags=re.IGNORECASE) for pattern in patterns)


def product_copy_signal_count(review: str) -> int:
    patterns = (
        r"(?:함유|첨가|배합|추출물|유도체|순도\s*\d+|저분자|고함량)",
        r"(?:미백|주름|보습|진정|탄력|항산화|기능성).{0,12}(?:효과|도움|개선|케어)",
        r"(?:피부|두피|모발|관절|장).{0,18}(?:개선|보호|관리|케어|도움|진정|수분\s*공급)",
        r"(?:제품|성분|포뮬러|제형)의?\s*(?:특징|효능|기능|장점)",
        r"(?:적합|권장|추천)\s*(?:대상|피부|용도)|(?:사용|섭취)\s*(?:방법|목적)",
        r"(?:제공합니다|도움을\s*줍니다|효과적입니다|설계되었습니다|사용됩니다)",
        r"(?:진정시키|수분을\s*공급하|보습하|관리하|개선하)(?:는|면서)",
        r"(?:대표적인\s*)?(?:[가-힣A-Za-z0-9]+\s*){0,4}"
        r"(?:제품|마스크|마스크팩|크림|세럼|앰플|로션|화장품)입니다",
        r"^[^\n.!?]{2,80}(?:은|는)\s+[^\n.!?]{5,}",
    )
    return sum(bool(re.search(pattern, str(review), flags=re.IGNORECASE)) for pattern in patterns)


def has_strong_promotional_cta(review: str) -> bool:
    patterns = (
        r"무조건\s*(?:사세요|구매하세요|쟁이세요|쟁여)",
        r"(?:두|세|열|백)\s*번\s*사세요",
        r"(?:여러|몇)\s*개\s*(?:쟁여|사두|구매)",
        r"(?:품절|없어지기)\s*전에\s*(?:사|구매|쟁여)",
        r"(?:반드시|당장)\s*(?:사세요|구매하세요|쟁여)",
    )
    return any(re.search(pattern, str(review), flags=re.IGNORECASE) for pattern in patterns)


def classify_pattern(row: Mapping[str, str]) -> tuple[str, str]:
    """Return one primary pattern using review-first, conservative priority."""
    review = str(row.get("review", ""))
    evidence = " ".join(
        filter(
            None,
            (
                str(row.get("collection_reason", "")).strip(),
                str(row.get("재검수근거", "")).strip(),
            ),
        )
    )
    if has_clear_internal_repetition(review):
        return "REPETITION", "리뷰 내부에서 같은 문장이나 구문이 세 차례 이상 반복됩니다."
    if has_structured_information(review):
        return "STRUCTURED_INFO", "성분·효능·특징이 항목 또는 섹션 구조로 나열됩니다."

    personal = has_personal_experience(review)
    copy_signals = product_copy_signal_count(review)
    evidence_supports_copy = bool(
        re.search(r"상세\s*페이지|상품\s*설명|제품\s*소개|광고\s*카피|설명문", evidence)
    )
    if not personal and (copy_signals >= 2 or (copy_signals >= 1 and evidence_supports_copy)):
        return "PRODUCT_COPY", "개인 사용 경험보다 제품 성분·효능·기능 설명이 중심입니다."
    if not personal and has_strong_promotional_cta(review):
        return "PROMOTIONAL_CTA", "개인 경험은 부족하고 강한 구매 또는 쟁임 유도가 핵심입니다."
    return (
        "AMBIGUOUS_TEXT_STRONG",
        "강한 텍스트 패턴이 명확하지 않아 정상 소비자 표현과 겹칠 가능성이 있습니다.",
    )


def select_usable_text_strong(
    retained_by_text: Mapping[str, Mapping[str, Any]],
    diagnoses_by_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for master_row in retained_by_text.values():
        if int(master_row["label"]) != LABEL_MAPPING["SUSPICIOUS"]:
            continue
        master_id = str(master_row["master_id"])
        diagnosis = diagnoses_by_id.get(master_id)
        if diagnosis is None:
            raise ValueError(f"No diagnostic CSV row for usable master_id={master_id}")
        if normalize_text(diagnosis["review"]) != normalize_text(str(master_row["text"])):
            raise ValueError(f"Workbook/CSV review mismatch for master_id={master_id}")
        if diagnosis["diagnostic_type"] != "TEXT_STRONG":
            continue
        classification_input = dict(diagnosis)
        classification_input["review"] = str(master_row["text"])
        pattern, reason = classify_pattern(classification_input)
        selected.append(
            {
                "master_id": master_id,
                "review": str(master_row["text"]),
                "previous_diagnostic_type": "TEXT_STRONG",
                "p_text_pattern": pattern,
                "p_text_v2_positive_candidate": str(pattern in POSITIVE_PATTERNS),
                "reason": reason,
            }
        )
    return selected


def run_sanity_checks(rows: Sequence[Mapping[str, str]]) -> None:
    """Check known examples after classification; IDs never drive the result."""
    by_id = {row["master_id"]: row for row in rows}
    for master_id in ("472", "480", "1008", "1009", "1010"):
        row = by_id.get(master_id)
        if row is None:
            warnings.warn(f"sanity check 대상 master_id={master_id}가 usable TEXT_STRONG에 없습니다.")
        elif row["p_text_pattern"] != "REPETITION":
            warnings.warn(
                f"sanity check 불일치: master_id={master_id}, "
                f"expected=REPETITION, actual={row['p_text_pattern']}"
            )

    for master_id in ("57", "93", "414", "599", "626"):
        row = by_id.get(master_id)
        if row is None:
            warnings.warn(f"sanity check 대상 master_id={master_id}가 usable TEXT_STRONG에 없습니다.")
        elif row["p_text_pattern"] != "AMBIGUOUS_TEXT_STRONG":
            warnings.warn(
                f"sanity check 불일치: master_id={master_id}, "
                f"expected=AMBIGUOUS_TEXT_STRONG, actual={row['p_text_pattern']}"
            )

    for master_id, expected in {
        "241": "STRUCTURED_INFO",
        "245": "PRODUCT_COPY",
        "255": "PROMOTIONAL_CTA",
    }.items():
        row = by_id.get(master_id)
        if row is None:
            warnings.warn(f"sanity check 대상 master_id={master_id}가 usable TEXT_STRONG에 없습니다.")
        elif row["p_text_pattern"] != expected:
            warnings.warn(
                f"sanity check 불일치: master_id={master_id}, "
                f"expected={expected}, actual={row['p_text_pattern']}"
            )

    for row in rows:
        if "메디힐" in row["review"] and row["p_text_pattern"] not in {
            "PRODUCT_COPY",
            "STRUCTURED_INFO",
        }:
            warnings.warn(
                f"메디힐 설명형 sanity check 불일치: master_id={row['master_id']}, "
                f"actual={row['p_text_pattern']}"
            )


def write_csv(rows: Sequence[Mapping[str, str]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError as exc:
        raise SystemExit(f"기존 결과를 덮어쓰지 않습니다: {output}") from exc


def print_report(reviewed_count: int, rows: Sequence[Mapping[str, str]]) -> None:
    counts = Counter(row["p_text_pattern"] for row in rows)
    positive = sum(row["p_text_v2_positive_candidate"] == "True" for row in rows)
    print(f"reviewed TEXT_STRONG: {reviewed_count}")
    print(f"usable TEXT_STRONG: {len(rows)}")
    print("\n패턴별 개수")
    for pattern in PATTERN_ORDER:
        print(f"{pattern}: {counts[pattern]}")
    print(f"\np_text_v2_positive_candidate=True: {positive}")
    print(f"p_text_v2_positive_candidate=False: {len(rows) - positive}")

    print("\n주의: candidate 값은 P_text v2 학습 후보 여부이며 Ground Truth나")
    print("정상/조작 확정 판정이 아닙니다.")
    for row in rows:
        print(f"\nmaster_id: {row['master_id']}")
        print(f"pattern: {row['p_text_pattern']}")
        print(f"candidate: {row['p_text_v2_positive_candidate']}")
        print(f"reason: {row['reason']}")
        print(f"review: {row['review']}")


def main() -> None:
    args = parse_args()
    source_rows = read_review_master(args.data_path)
    prepared = prepare_records(source_rows)
    retained = prepare_master_rows(source_rows, prepared.examples)
    diagnoses = load_diagnoses(args.diagnosis_path)

    reviewed_count = sum(
        row["diagnostic_type"] == "TEXT_STRONG" for row in diagnoses.values()
    )
    rows = select_usable_text_strong(retained, diagnoses)
    if reviewed_count != EXPECTED_REVIEWED_TEXT_STRONG:
        raise SystemExit(
            f"Expected {EXPECTED_REVIEWED_TEXT_STRONG} reviewed TEXT_STRONG rows, "
            f"found {reviewed_count}; no output was written."
        )
    if len(rows) != EXPECTED_USABLE_TEXT_STRONG:
        raise SystemExit(
            f"Expected {EXPECTED_USABLE_TEXT_STRONG} usable TEXT_STRONG rows, "
            f"found {len(rows)}; no output was written."
        )

    run_sanity_checks(rows)
    print_report(reviewed_count, rows)
    write_csv(rows, args.output_path)
    print(f"\nCSV created: {args.output_path}")


if __name__ == "__main__":
    main()
