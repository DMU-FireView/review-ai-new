from copy import deepcopy

from app.training.p_text.build_contrastive_pilot_v1 import build_rows
from app.training.p_text.contrastive_pilot import validate_pilot
from app.training.p_text.contrastive_pilot import (
    ForbiddenRecord,
    cross_corpus_near_template_matches,
    discover_forbidden_paths,
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
