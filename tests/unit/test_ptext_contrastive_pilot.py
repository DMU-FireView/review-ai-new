from copy import deepcopy

from app.training.p_text.build_contrastive_pilot_v1 import build_rows
from app.training.p_text.contrastive_pilot import validate_pilot
from app.training.p_text.contrastive_pilot import (
    ForbiddenRecord,
    cross_corpus_near_template_matches,
    discover_forbidden_paths,
    health_keyword_balance,
)


def _valid_rows():
    return build_rows()


def test_valid_pilot_skeleton_passes() -> None:
    report = validate_pilot(_valid_rows())
    assert report.passed, report.errors
    assert report.counts["pairs"] == 50
    assert report.counts["pending_human_pairs"] == 30
    assert report.counts["synthetic_or_augmented_pairs"] == 20
    assert report.counts["training_ready_rows"] == 0


def test_pair_label_error_fails() -> None:
    rows = _valid_rows()
    rows[1]["label"] = "SUSPICIOUS"
    assert any("one NORMAL and one SUSPICIOUS" in error for error in validate_pilot(rows).errors)


def test_duplicate_sample_id_fails() -> None:
    rows = _valid_rows()
    rows[1]["sample_id"] = rows[0]["sample_id"]
    assert any("duplicate sample_id" in error for error in validate_pilot(rows).errors)


def test_blank_human_placeholder_cannot_be_training_ready() -> None:
    rows = _valid_rows()
    placeholder = next(row for row in rows if row["source_type"] == "PENDING_HUMAN")
    placeholder["review_status"] = "AGREED"
    assert any("blank placeholder marked training-ready" in error for error in validate_pilot(rows).errors)


def test_exact_duplicate_fails() -> None:
    rows = _valid_rows()
    rows[2]["content"] = rows[0]["content"]
    assert any("exact duplicate groups" in error for error in validate_pilot(rows).errors)


def test_label_conflict_fails() -> None:
    rows = _valid_rows()
    normal = next(row for row in rows if row["label"] == "NORMAL" and row["content"])
    suspicious = next(row for row in rows if row["label"] == "SUSPICIOUS" and row["parent_pair_id"] != normal["parent_pair_id"])
    suspicious["content"] = normal["content"].swapcase()
    report = validate_pilot(rows)
    assert any("label conflict groups" in error for error in report.errors)


def test_generation_family_limit_fails() -> None:
    rows = _valid_rows()
    populated_pair_ids = []
    for row in rows:
        if row["content"] and row["parent_pair_id"] not in populated_pair_ids:
            populated_pair_ids.append(row["parent_pair_id"])
    for row in rows:
        if row["parent_pair_id"] in populated_pair_ids[:5]:
            row["generation_family_id"] = "overused-family"
    assert any("generation family limit exceeded" in error for error in validate_pilot(rows).errors)


def test_keyword_only_minimal_pair_fails() -> None:
    rows = _valid_rows()
    pair_id = rows[0]["parent_pair_id"]
    pair = [row for row in rows if row["parent_pair_id"] == pair_id]
    pair[0]["content"] = "이 제품을 추천합니다"
    pair[1]["content"] = "이 제품을 비추천합니다"
    assert any("keyword-only minimal pair" in error for error in validate_pilot(rows).errors)


def test_forbidden_overlap_fails() -> None:
    rows = _valid_rows()
    forbidden = [rows[0]["content"]]
    report = validate_pilot(rows, forbidden_contents=forbidden)
    assert report.counts["forbidden_exact_overlap"] == 1
    assert not report.passed


def test_cross_corpus_unrelated_text_passes() -> None:
    rows = _valid_rows()
    forbidden = [ForbiddenRecord("challenge.xlsx", "review_id:x1", "겨울 산책 중 장갑 안쪽 봉제가 풀렸습니다.", "challenge")]
    assert cross_corpus_near_template_matches(rows, forbidden) == []


def test_cross_corpus_exact_duplicate_is_candidate_and_preserves_source_identifier() -> None:
    rows = _valid_rows()
    forbidden = [ForbiddenRecord("blind.xlsx", "blind_id:B-1", rows[0]["content"], "blind")]
    report = validate_pilot(rows, forbidden_records=forbidden)
    match = report.cross_corpus_matches[0]
    assert match.similarity == 1.0
    assert match.candidate is True
    assert match.forbidden_source == "blind.xlsx"
    assert match.forbidden_row_identifier == "blind_id:B-1"
    assert not report.passed


def test_cross_corpus_normalized_duplicate_is_candidate() -> None:
    rows = _valid_rows()
    content = rows[0]["content"]
    forbidden = [ForbiddenRecord("challenge.xlsx", "row:2", f"  {content.upper()}  ", "challenge")]
    report = validate_pilot(rows, forbidden_records=forbidden)
    assert report.counts["forbidden_normalized_overlap"] == 1
    assert report.cross_corpus_matches[0].candidate is True


def test_cross_corpus_near_template_at_threshold_is_candidate() -> None:
    rows = _valid_rows()
    content = rows[0]["content"]
    near = content.replace("크림이면", "제품이면")
    forbidden = [ForbiddenRecord("training.xlsx", "review_id:T-1", near, "training")]
    matches = cross_corpus_near_template_matches(rows, forbidden)
    assert matches
    assert matches[0].similarity >= 0.85
    assert matches[0].candidate is True


def test_discovery_excludes_pilot_tree(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    corpus = tmp_path / "corpus"
    pilot.mkdir()
    corpus.mkdir()
    (pilot / "annotation.jsonl").write_text("{}\n", encoding="utf-8")
    expected = corpus / "training.jsonl"
    expected.write_text("{}\n", encoding="utf-8")
    assert discover_forbidden_paths([tmp_path], pilot_root=pilot) == [expected]


def _extended_pair(*, category="EXAGGERATED_PRAISE_WITHOUT_EVIDENCE", domain="cosmetics"):
    rows = deepcopy(_valid_rows()[:2])
    for row in rows:
        row["category"] = category
        row["domain"] = domain
        row["praise_intensity"] = "extreme"
        row["evidence_level"] = "none" if row["label"] == "SUSPICIOUS" else "sufficient"
        row["human_reviewed"] = False
        row["human_approved"] = False
    return rows


def _validate_extended(rows):
    return validate_pilot(rows, expected_pairs_per_category=None)


def test_exaggerated_praise_category_passes() -> None:
    assert _validate_extended(_extended_pair()).passed


def test_cosmetics_domain_passes() -> None:
    assert _validate_extended(_extended_pair(domain="cosmetics")).passed


def test_health_supplement_domain_passes() -> None:
    assert _validate_extended(_extended_pair(domain="health_supplement")).passed


def test_pharma_otc_domain_passes() -> None:
    assert _validate_extended(_extended_pair(domain="pharma_otc")).passed


def test_invalid_praise_intensity_fails() -> None:
    rows = _extended_pair()
    rows[0]["praise_intensity"] = "maximum"
    assert any("invalid praise_intensity" in error for error in _validate_extended(rows).errors)


def test_invalid_evidence_level_fails() -> None:
    rows = _extended_pair()
    rows[0]["evidence_level"] = "unknown"
    assert any("invalid evidence_level" in error for error in _validate_extended(rows).errors)


def test_synthetic_llm_generated_passes() -> None:
    rows = _extended_pair()
    assert all(row["source_type"] == "synthetic" and row["is_llm_generated"] for row in rows)
    assert _validate_extended(rows).passed


def test_synthetic_human_reviewed_and_approved_passes() -> None:
    rows = _extended_pair()
    for row in rows:
        row["human_reviewed"] = True
        row["human_approved"] = True
    assert _validate_extended(rows).passed


def test_llm_generated_human_written_conflict_fails() -> None:
    rows = _extended_pair()
    for row in rows:
        row["source_type"] = "human_written"
    assert any("conflicts with source_type=human_written" in error for error in _validate_extended(rows).errors)


def test_extreme_praise_with_sufficient_evidence_normal_passes() -> None:
    rows = _extended_pair()
    normal = next(row for row in rows if row["label"] == "NORMAL")
    assert normal["praise_intensity"] == "extreme"
    assert normal["evidence_level"] == "sufficient"
    assert _validate_extended(rows).passed


def test_extreme_praise_without_evidence_suspicious_passes() -> None:
    rows = _extended_pair()
    suspicious = next(row for row in rows if row["label"] == "SUSPICIOUS")
    assert suspicious["praise_intensity"] == "extreme"
    assert suspicious["evidence_level"] == "none"
    assert _validate_extended(rows).passed


def test_health_keyword_normal_sample_passes() -> None:
    rows = _extended_pair(domain="health_supplement")
    normal = next(row for row in rows if row["label"] == "NORMAL")
    normal["content"] = "비타민을 세 달 복용했고 피로에는 변화가 없지만 크기가 작아 만족합니다."
    assert _validate_extended(rows).passed


def test_health_keyword_suspicious_sample_passes() -> None:
    rows = _extended_pair(domain="health_supplement")
    suspicious = next(row for row in rows if row["label"] == "SUSPICIOUS")
    suspicious["content"] = "비타민 성분만 봤고 복용 전이지만 피로 개선 효과가 완벽하다고 확신합니다."
    assert _validate_extended(rows).passed


def test_health_keyword_balance_reports_both_labels() -> None:
    rows = _extended_pair(domain="health_supplement")
    rows[0]["content"] = "복용 전이지만 이 비타민 효과는 완벽합니다."
    rows[1]["content"] = "이 비타민을 세 달 복용했지만 효과는 잘 모르겠습니다."
    balance = health_keyword_balance(rows)
    assert balance["비타민"] == {"NORMAL": 1, "SUSPICIOUS": 1}
    assert balance["효과"] == {"NORMAL": 1, "SUSPICIOUS": 1}


def test_short_health_keyword_does_not_match_inside_unrelated_word() -> None:
    rows = _extended_pair()
    rows[0]["content"] = "배송 날짜를 예약했지만 제품은 아직 사용하지 않았습니다."
    rows[1]["content"] = "세 달 동안 사용한 뒤 만족해서 약을 보관하는 데 쓰고 있습니다."
    balance = health_keyword_balance(rows)
    assert balance["약"] == {"NORMAL": 1, "SUSPICIOUS": 0}


def test_legacy_pilot_row_without_extended_metadata_passes() -> None:
    rows = _valid_rows()
    for row in rows:
        for name in ("praise_intensity", "evidence_level", "human_reviewed", "human_approved"):
            row.pop(name, None)
    assert validate_pilot(rows).passed


def test_pending_human_placeholder_remains_protected_with_extended_metadata() -> None:
    rows = _valid_rows()
    placeholder = next(row for row in rows if row["source_type"] == "PENDING_HUMAN")
    placeholder["human_reviewed"] = True
    placeholder["human_approved"] = True
    placeholder["review_status"] = "AGREED"
    assert any("blank placeholder marked training-ready" in error for error in validate_pilot(rows).errors)
