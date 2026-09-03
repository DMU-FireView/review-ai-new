import pytest

from app.training.p_text.data import (
    EXCLUDED_LABELS, LABEL_MAPPING, InsufficientDataError,
    compute_class_weights, prepare_records, split_counts, stratified_split,
)
from app.training.p_text.metrics import classification_metrics
from app.training.p_text.train import Config


def test_label_mapping_and_excluded_labels() -> None:
    assert LABEL_MAPPING == {"NORMAL": 0, "SUSPICIOUS": 1}
    assert EXCLUDED_LABELS == {"UNCERTAIN", "DISCLOSED_PROMO", "INVALID"}


def test_default_checkpoint_is_koelectra() -> None:
    # This is a configuration-only test: it must never download the model.
    assert Config.model_name == "monologg/koelectra-base-v3-discriminator"


def test_filtering_and_text_only_model_input() -> None:
    rows = [
        {"review": "normal", "final_label": "NORMAL", "platform": "x", "collection_reason": "human"},
        {"review": "sus", "final_label": "SUSPICIOUS", "source_group": "human"},
        {"review": "skip", "final_label": "UNCERTAIN"},
        {"review": "promo", "final_label": "DISCLOSED_PROMO"},
        {"review": "invalid", "final_label": "INVALID"},
    ]
    prepared = prepare_records(rows)
    assert prepared.examples == [{"text": "normal", "label": 0}, {"text": "sus", "label": 1}]
    assert all(set(row) == {"text", "label"} for row in prepared.examples)


def test_conflicting_duplicates_are_removed_before_split() -> None:
    prepared = prepare_records([
        {"review": " same\ntext ", "final_label": "NORMAL"},
        {"review": "same text", "final_label": "SUSPICIOUS"},
        {"review": "unique", "final_label": "NORMAL"},
    ])
    assert prepared.examples == [{"text": "unique", "label": 0}]
    assert prepared.duplicate_rows_removed == 2
    assert prepared.conflicting_groups_removed == 1


def test_stratified_split_is_reproducible_and_preserves_classes() -> None:
    rows = [{"text": f"n{i}", "label": 0} for i in range(20)] + [{"text": f"s{i}", "label": 1} for i in range(4)]
    first = stratified_split(rows, seed=42)
    assert first == stratified_split(rows, seed=42)
    assert split_counts(first) == {
        "train": {"NORMAL": 16, "SUSPICIOUS": 2},
        "validation": {"NORMAL": 2, "SUSPICIOUS": 1},
        "test": {"NORMAL": 2, "SUSPICIOUS": 1},
    }


def test_class_weights_use_training_distribution() -> None:
    rows = [{"text": str(i), "label": 0} for i in range(8)] + [{"text": "p", "label": 1}]
    assert compute_class_weights(rows) == pytest.approx([9 / 16, 9 / 2])


def test_metrics_include_suspicious_and_confusion_matrix() -> None:
    result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert result["accuracy"] == 0.5
    assert result["suspicious"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_too_few_minority_examples_fails_safely() -> None:
    rows = [{"text": f"n{i}", "label": 0} for i in range(10)] + [{"text": "s0", "label": 1}, {"text": "s1", "label": 1}]
    with pytest.raises(InsufficientDataError, match="at least 3 unique examples"):
        stratified_split(rows)
