from copy import deepcopy
from unittest.mock import patch

import pytest

from app.training.p_text import train
from app.training.p_text.data import (
    InsufficientDataError, compute_class_weights, prepare_records, split_counts,
    stratified_split,
)


@pytest.fixture
def real_prepared():
    return prepare_records([
        {"review": f"Real normal {i}", "final_label": "NORMAL"} for i in range(10)
    ] + [
        {"review": f"Real suspicious {i}", "final_label": "SUSPICIOUS"} for i in range(4)
    ])


def synthetic(text, label="SUSPICIOUS", flag="YES"):
    return {
        "content": text, "final_label": label, "use_for_training": flag,
        "synthetic_id": "s1", "source_set": "synthetic", "risk_type": "test",
        "split_policy": "TRAIN_ONLY", "generation_note": "audit only",
    }


def test_cli_augmentation_default_and_explicit_path(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"])
    assert train.Config().train_augmentation_path is None
    assert train.parse_args().train_augmentation_path is None
    monkeypatch.setattr("sys.argv", [
        "train", "--data-format", "v2", "--train-augmentation-path", "synthetic.xlsx",
    ])
    assert train.parse_args().train_augmentation_path == "synthetic.xlsx"


@pytest.mark.parametrize("data_format", ["baseline", "v2"])
def test_disabled_preserves_splits_and_weights(real_prepared, data_format):
    expected = stratified_split(real_prepared.examples, seed=42)
    with patch.object(train, "read_ptext_v2_workbook") as loader:
        splits, audit = train.prepare_training_splits(train.Config(data_format=data_format), real_prepared)
    loader.assert_not_called()
    assert splits == expected
    assert compute_class_weights(splits["train"]) == compute_class_weights(expected["train"])
    assert audit == {
        "augmentation_enabled": False, "augmentation_path": None,
        "augmentation_raw_selected": 0, "augmentation_used_after_deduplication": 0,
        "augmentation_real_collision_removed": 0, "augmentation_added_to_train": 0,
        "real_split_counts": split_counts(expected), "final_split_counts": split_counts(expected),
    }


def test_train_only_collisions_dedup_metadata_and_audit(real_prepared):
    original_prepared = deepcopy(real_prepared)
    real_splits = stratified_split(real_prepared.examples, seed=42)
    original_splits = deepcopy(real_splits)
    rows = [
        synthetic(" New\nsynthetic "), synthetic("new SYNTHETIC"),
        synthetic("second unique"), synthetic("ignored", flag="NO"),
        synthetic("uncertain", label="UNCERTAIN"),
    ] + [
        synthetic("  " + real_splits[name][0]["text"].upper().replace(" ", "\t") + "  ")
        for name in ("train", "validation", "test")
    ]
    events = []

    def split(examples, *, seed):
        events.append("real split")
        assert examples == original_prepared.examples
        assert seed == 42
        return real_splits

    def load(path):
        assert events == ["real split"]
        events.append("synthetic load")
        return rows

    config = train.Config(data_format="v2", train_augmentation_path="synthetic.xlsx")
    with (
        patch.object(train, "stratified_split", side_effect=split) as splitter,
        patch.object(train, "read_ptext_v2_workbook", side_effect=load) as loader,
    ):
        splits, audit = train.prepare_training_splits(config, real_prepared)
    splitter.assert_called_once()
    loader.assert_called_once_with("synthetic.xlsx")
    assert real_prepared == original_prepared
    for name in ("validation", "test"):
        assert splits[name] is real_splits[name]
        assert splits[name] == original_splits[name]
    assert splits["train"] == original_splits["train"] + [
        {"text": "New\nsynthetic", "label": 1}, {"text": "second unique", "label": 1},
    ]
    assert all(set(row) == {"text", "label"} for rows in splits.values() for row in rows)
    assert audit == {
        "augmentation_enabled": True, "augmentation_path": "synthetic.xlsx",
        "augmentation_raw_selected": 6, "augmentation_used_after_deduplication": 5,
        "augmentation_real_collision_removed": 3, "augmentation_added_to_train": 2,
        "real_split_counts": split_counts(original_splits), "final_split_counts": split_counts(splits),
    }
    assert audit["real_split_counts"]["train"] == {"NORMAL": 8, "SUSPICIOUS": 2}
    assert audit["final_split_counts"]["train"] == {"NORMAL": 8, "SUSPICIOUS": 4}
    assert compute_class_weights(splits["train"]) == pytest.approx([.75, 1.5])
    assert compute_class_weights(original_splits["train"]) == pytest.approx([.625, 2.5])


@pytest.mark.parametrize("rows", [
    [synthetic("normal", label="NORMAL")],
    [synthetic("same"), synthetic(" SAME ", label=" normal "), synthetic("unique")],
    [synthetic("normal", label="NORMAL", flag="NO"), synthetic("unique")],
])
def test_normal_is_rejected_before_conflict_removal(real_prepared, rows):
    real_splits = stratified_split(real_prepared.examples)
    snapshot = deepcopy(real_splits)
    with (
        patch.object(train, "read_ptext_v2_workbook", return_value=rows),
        patch.object(train, "stratified_split", return_value=real_splits),
        pytest.raises(ValueError, match="found NORMAL"),
    ):
        train.prepare_training_splits(train.Config(train_augmentation_path="synthetic.xlsx"), real_prepared)
    assert real_splits == snapshot


@pytest.mark.parametrize("rows", [
    [], [synthetic(" \n ")], [synthetic(None)],
    [synthetic("ignored", flag="NO")], [synthetic("invalid", label="INVALID")],
])
def test_empty_usable_synthetic_is_rejected(real_prepared, rows):
    with patch.object(train, "read_ptext_v2_workbook", return_value=rows):
        with pytest.raises(ValueError, match="no usable SUSPICIOUS"):
            train.prepare_training_splits(train.Config(train_augmentation_path="synthetic.xlsx"), real_prepared)


def test_all_collisions_add_zero_without_replication(real_prepared):
    expected = stratified_split(real_prepared.examples)
    rows = [synthetic(row["text"]) for row in real_prepared.examples]
    with patch.object(train, "read_ptext_v2_workbook", return_value=rows):
        splits, audit = train.prepare_training_splits(
            train.Config(train_augmentation_path="synthetic.xlsx"), real_prepared,
        )
    assert splits == expected
    assert audit["augmentation_real_collision_removed"] == len(rows)
    assert audit["augmentation_added_to_train"] == 0


def test_synthetic_cannot_bypass_real_minimum_class_size():
    insufficient = prepare_records([
        {"review": f"normal {i}", "final_label": "NORMAL"} for i in range(3)
    ] + [{"review": "positive", "final_label": "SUSPICIOUS"}])
    with patch.object(train, "read_ptext_v2_workbook") as loader:
        with pytest.raises(InsufficientDataError, match="at least 3 unique examples"):
            train.prepare_training_splits(train.Config(train_augmentation_path="synthetic.xlsx"), insufficient)
    loader.assert_not_called()
