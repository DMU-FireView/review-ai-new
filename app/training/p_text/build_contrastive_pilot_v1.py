"""Build the v1 contrastive annotation pilot; no model or training imports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.p_text.contrastive_pilot import CATEGORIES, EXPORT_COLUMNS


OUTPUT_DIR = Path("data/ptext_improvement/pilot_v1")
JSONL_PATH = OUTPUT_DIR / "annotation_pilot_v1.jsonl"
CSV_PATH = OUTPUT_DIR / "annotation_pilot_v1.csv"
CREATED_AT = "2026-10-03"
REVISED_PAIR_IDS = frozenset(
    {
        "pilot-v1-p009", "pilot-v1-p010", "pilot-v1-p011", "pilot-v1-p012",
        "pilot-v1-p017", "pilot-v1-p018", "pilot-v1-p019", "pilot-v1-p020",
    }
)


# Each tuple is category, domain, difficulty, length bucket, style, relation,
# suspicious content, matched-normal content.
SYNTHETIC_PAIRS = (
    ("EXPERIENCE_FREE_CLAIM", "cosmetics_health", "easy", "short", "concise",
     "unverified_effect_certainty_vs_explicit_post_use_verification",
     "뚜껑도 열기 전인데 이 크림이면 붉은 기가 금방 가라앉을 게 분명해요.",
     "뚜껑을 열어 향과 발림만 확인했어요. 붉은 기가 줄어드는지는 며칠 더 써보려고 합니다."),
    ("EXPERIENCE_FREE_CLAIM", "food", "medium", "medium", "polite",
     "nutrition_description_as_certain_result_vs_uncertain_expectation",
     "아직 맛보지는 않았지만 원료 구성을 보니 아침마다 먹으면 오후 피로가 확실히 줄겠습니다.",
     "아직 맛보지는 않았지만 원료 구성은 괜찮아 보입니다. 오후 피로에 도움이 되는지는 먹어 본 뒤 판단하겠습니다."),
    ("EXPERIENCE_FREE_CLAIM", "electronics", "hard", "long", "conversational",
     "specification_based_lifetime_claim_vs_observation_plan",
     "설치 날짜가 남아서 전원도 못 켰어요. 그래도 소비 전력과 모터 설명을 보니 밤새 돌려도 소음 걱정 없이 몇 년은 거뜬하겠네요.",
     "설치 날짜가 남아서 전원은 아직 못 켰어요. 소비 전력과 모터 설명은 마음에 들지만 밤 소음과 내구성은 직접 확인해 보려고요."),
    ("EXPERIENCE_FREE_CLAIM", "household", "medium", "medium", "polite",
     "appearance_based_durability_certainty_vs_limited_preuse_impression",
     "상자 안에서 재질만 봤는데 이 수납함은 습한 곳에서도 휘지 않고 오래 버틸 품질입니다.",
     "상자 안에서 재질을 살펴보니 마감은 단단해 보입니다. 습한 곳에서 휘는지는 실제로 두고 확인해야 합니다."),

    ("UNUSED_RECOMMENDATION", "fashion", "easy", "short", "conversational",
     "unused_universal_recommendation_vs_try_on_before_recommendation",
     "아직 입어 보진 않았는데 체형 상관없이 잘 맞을 옷이라 주변에도 권했어요.",
     "아직 입어 보진 않았고 색만 확인했어요. 핏을 본 다음 비슷한 체형인 친구에게 알려 주려고요."),
    ("UNUSED_RECOMMENDATION", "electronics", "medium", "medium", "polite",
     "manual_only_purchase_nudge_vs_conditional_information_sharing",
     "설명서만 읽어 봤지만 조작이 쉬워 부모님 댁에도 바로 주문해 드려도 되겠습니다.",
     "설명서의 버튼 구성은 단순해 보입니다. 직접 조작해 본 뒤 부모님이 쓰기 편한지 말씀드리겠습니다."),
    ("UNUSED_RECOMMENDATION", "food", "hard", "long", "polite",
     "untried_gift_recommendation_vs_recipient_feedback_requirement",
     "포장만 보고 냉장고에 넣어 두었습니다. 맛과 향이 누구에게나 부담 없을 구성이라 선물용을 찾는 분께 권하고 싶습니다.",
     "포장 상태는 깔끔해서 냉장고에 넣어 두었습니다. 맛과 향은 먹어 본 가족의 반응을 확인한 뒤 선물용으로 권할지 정하겠습니다."),
    ("UNUSED_RECOMMENDATION", "cosmetics_health", "medium", "medium", "conversational",
     "ingredient_list_purchase_advice_vs_patch_test_first",
     "성분표만 확인했는데 민감한 분도 걱정 없이 고를 수 있겠어요. 고민 중인 동료에게 이걸 사라고 했습니다.",
     "성분표에는 피하고 싶은 향료가 없네요. 제 피부에 먼저 시험한 뒤 민감한 동료에게 정보를 전하려고요."),

    ("NATURAL_PROMOTIONAL", "household", "medium", "medium", "conversational",
     "lifestyle_story_purchase_nudge_vs_personal_choice_without_nudge",
     "정리할 시간이 부족해서 이 수납함은 칸 구성만 잠깐 살펴봤어요. 아직 물건을 넣어 보진 않았지만 집안일을 줄이고 싶다면 더 비교할 필요 없이 꼭 필요한 선택입니다.",
     "정리할 시간이 부족해서 이 수납함에 주방 도구를 일주일 동안 넣어 봤어요. 제 물건 수에는 칸 구성이 잘 맞지만 큰 냄비가 많은 집에는 공간이 부족할 수 있습니다."),
    ("NATURAL_PROMOTIONAL", "fashion", "hard", "long", "polite",
     "subtle_wardrobe_nudge_vs_experience_bounded_preference",
     "매장에서 옷걸이에 걸린 모습만 보고 아직 입어 보지는 않았어요. 그래도 이 디자인은 계절과 자리를 가리지 않아 한 벌 마련해 두면 누구나 옷 고르기가 쉬워질 것 같습니다.",
     "옷을 고를 때 손이 자주 가는지가 중요해서 일주일 동안 세 번 입어 봤습니다. 제 출근복에는 잘 맞았지만 격식을 차리는 자리에는 캐주얼해 보입니다."),
    ("NATURAL_PROMOTIONAL", "food", "easy", "short", "concise",
     "soft_cart_nudge_vs_personal_taste_boundary",
     "포장에 적힌 맛 설명과 간식 구성만 봤고 아직 먹어 보진 않았어요. 그래도 단맛이 누구에게나 알맞을 테니 간식 고민 없이 장바구니에 담아도 됩니다.",
     "이 간식을 직접 먹어 보니 단맛이 약하고 구성은 간단해서 제 입에는 잘 맞았어요. 진한 맛이나 다양한 식감을 좋아하면 심심하게 느낄 수 있습니다."),
    ("NATURAL_PROMOTIONAL", "electronics", "hard", "medium", "polite",
     "convenience_promise_as_purchase_reason_vs_observed_limit",
     "전원을 잠깐 켜서 기본 화면만 확인했고 세부 기능은 아직 사용하지 않았어요. 그래도 조작이 누구에게나 편할 테니 더 비교하지 않고 선택해도 될 전자제품입니다.",
     "이 전자제품을 사흘 동안 사용해 보니 기본 화면과 자주 쓰는 기능은 조작하기 편했습니다. 세부 설정과 장시간 사용 때의 안정성은 더 확인해야 합니다."),

    ("PROMOTION_EXPERIENCE_MIX", "electronics", "easy", "short", "concise",
     "single_observation_to_universal_lifetime_claim_vs_limited_first_use",
     "한 번 켜 보니 반응이 빠르네요. 누구나 오래 고장 걱정 없이 쓸 기기입니다.",
     "한 번 켜 보니 반응은 빨랐어요. 발열과 배터리는 더 사용해 봐야 알겠습니다."),
    ("PROMOTION_EXPERIENCE_MIX", "household", "medium", "medium", "conversational",
     "packaging_impression_to_total_convenience_vs_observed_packaging_only",
     "포장이 단단하게 와서 믿음이 갔어요. 이 정도면 설치부터 관리까지 어떤 집에서도 번거로울 일이 없겠네요.",
     "포장이 단단하게 와서 파손은 없었어요. 설치와 관리가 편한지는 주말에 조립해 보고 적으려고요."),
    ("PROMOTION_EXPERIENCE_MIX", "cosmetics_health", "hard", "long", "polite",
     "brief_texture_test_to_broad_skin_outcome_vs_bounded_observation",
     "손등에 한 번 펴 발랐는데 흡수가 빠르고 산뜻했습니다. 꾸준히 바르면 피부 타입과 관계없이 유수분 균형이 안정될 제품이라고 봅니다.",
     "손등에 한 번 펴 발랐는데 흡수가 빠르고 산뜻했습니다. 얼굴에 사용할 때 자극이나 보습 지속력이 어떤지는 아직 판단하기 어렵습니다."),
    ("PROMOTION_EXPERIENCE_MIX", "fashion", "medium", "medium", "polite",
     "minor_fit_check_to_universal_recommendation_vs_body_specific_result",
     "소매 길이만 확인했는데 실루엣이 깔끔합니다. 체형을 타지 않아 옷 고르기 어려운 분도 만족하겠습니다.",
     "소매 길이와 어깨선을 확인하니 제 체형에는 잘 맞습니다. 다른 체형에서도 같은 핏인지는 모르겠습니다."),

    ("FEATURE_AS_PERSONAL_EXPERIENCE", "food", "easy", "short", "concise",
     "package_feature_as_consumed_effect_vs_actual_taste_observation",
     "포장의 영양정보만 읽었고 아직 먹지는 않았어요. 그래도 이 균형 잡힌 영양 구성이면 먹는 날부터 하루 컨디션이 확실히 가벼워집니다.",
     "이 제품을 아침에 사흘 동안 먹어 보니 점심 전 허기는 덜했어요. 하루 컨디션 전체가 달라졌는지는 아직 판단하기 어렵습니다."),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "cosmetics_health", "medium", "medium", "polite",
     "ingredient_benefits_as_personal_result_vs_observed_skin_response",
     "진정 성분과 보습 성분만 확인했고 얼굴에는 아직 사용하지 않았어요. 그래도 바르면 피부 속 균형이 바로 채워져 하루 종일 편안함이 이어집니다.",
     "이 제품을 저녁마다 일주일 사용하니 바른 직후 당김은 줄었습니다. 하루 종일 보습이 유지되는지와 붉은 기 변화는 아직 확인하지 못했습니다."),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "household", "hard", "long", "conversational",
     "catalog_features_as_lived_convenience_vs_specific_use_evidence",
     "넓은 입구와 분리 구조만 살펴봤고 아직 물건을 넣거나 씻어 보지는 않았어요. 그래도 이 구조면 설거지부터 보관까지 제 생활을 알아서 정돈해 매일의 번거로움이 사라집니다.",
     "넓은 입구라 수세미가 안쪽까지 닿았고 분리 부품도 물로 쉽게 씻겼어요. 다만 건조할 때는 부품을 따로 펼쳐 둘 공간이 필요했습니다."),
    ("FEATURE_AS_PERSONAL_EXPERIENCE", "fashion", "hard", "long", "polite",
     "material_description_as_all_day_result_vs_measured_wear_experience",
     "원단과 밑창을 손으로 잠깐 확인했을 뿐 아직 신어 보지는 않았어요. 그래도 가벼운 원단과 유연한 밑창이 아침부터 저녁까지 발의 피로를 확실히 덜어 줍니다.",
     "이 신발을 출근길과 점심시간에 합쳐 세 시간 정도 신었습니다. 원단은 가볍고 밑창은 유연했지만 발볼은 오후에 조금 조였습니다."),
)


def _row(
    *, pair_number: int, label: str, category: str, domain: str,
    difficulty: str, length_bucket: str, style: str, relation: str,
    content: str, synthetic: bool,
) -> dict[str, object]:
    suffix = "s" if label == "SUSPICIOUS" else "n"
    pair_id = f"pilot-v1-p{pair_number:03d}"
    return {
        "sample_id": f"{pair_id}-{suffix}",
        "content": content,
        "label": label,
        "category": category,
        "secondary_categories": "",
        "domain": domain,
        "source_type": "synthetic" if synthetic else "PENDING_HUMAN",
        "parent_pair_id": pair_id,
        "generation_family_id": f"pilot-v1-family-{pair_number:03d}",
        "difficulty": difficulty,
        "review_status": "DRAFT",
        "notes": (
            f"{relation}; revision_reason=text_observable_label"
            if synthetic and pair_id in REVISED_PAIR_IDS
            else relation if synthetic
            else "TODO: human author and two reviewers must complete"
        ),
        "author_id": "codex" if synthetic else "",
        "reviewer_ids": "",
        "split": "",
        "seed": "",
        "created_at": CREATED_AT,
        "source_reference": "",
        "creation_method": "codex_generated_candidate" if synthetic else "human_placeholder",
        "original_source_type": "",
        "is_llm_generated": synthetic,
        "pair_relation": relation if synthetic else "TODO",
        "length_bucket": length_bucket if synthetic else "TODO",
        "style": style if synthetic else "TODO",
        "praise_intensity": "",
        "evidence_level": "",
        "human_reviewed": False,
        "human_approved": False,
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pair_number = 1
    for category, domain, difficulty, length_bucket, style, relation, suspicious, normal in SYNTHETIC_PAIRS:
        rows.append(_row(pair_number=pair_number, label="SUSPICIOUS", category=category,
                         domain=domain, difficulty=difficulty, length_bucket=length_bucket,
                         style=style, relation=relation, content=suspicious, synthetic=True))
        rows.append(_row(pair_number=pair_number, label="NORMAL", category=category,
                         domain=domain, difficulty=difficulty, length_bucket=length_bucket,
                         style=style, relation=relation, content=normal, synthetic=True))
        pair_number += 1

    placeholder_domains = ("cosmetics_health", "food", "fashion", "household", "electronics", "mixed")
    placeholder_difficulties = ("easy", "medium", "hard", "medium", "hard", "easy")
    for category in CATEGORIES:
        for index in range(6):
            for label in ("SUSPICIOUS", "NORMAL"):
                rows.append(_row(pair_number=pair_number, label=label, category=category,
                                 domain=placeholder_domains[index],
                                 difficulty=placeholder_difficulties[index], length_bucket="",
                                 style="", relation="", content="", synthetic=False))
            pair_number += 1
    return rows


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with JSONL_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with CSV_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
