from dataclasses import asdict
import json
from unittest.mock import patch

import pytest

from app.training.p_text import train
from app.training.p_text.data import prepare_records


def test_default_cli_preserves_baseline_configuration(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train"])
    config = train.parse_args()
    assert config == train.Config()
    assert config.data_format == "baseline"
    assert config.data_path == "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
    assert config.output_dir == "artifacts/p_text_baseline"
    assert config.model_name == "monologg/koelectra-base-v3-discriminator"
    assert (config.epochs, config.batch_size, config.learning_rate, config.max_length, config.seed) == (
        3, 8, 2e-5, 256, 42,
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_baseline_uses_original_loader_and_default_preparation(explicit):
    config = train.Config(data_format="baseline") if explicit else train.Config()
    rows = [
        {"review": "normal", "final_label": "NORMAL", "use_for_training": "NO"},
        {"review": "suspicious", "final_label": "SUSPICIOUS", "source_note": "audit"},
        {"review": "excluded", "final_label": "INVALID"},
    ]
    with (
        patch.object(train, "read_review_master", return_value=rows) as baseline,
        patch.object(train, "read_ptext_v2_workbook") as v2,
        patch.object(train, "prepare_records", wraps=prepare_records) as prepare,
    ):
        prepared = train.load_prepared_data(config)
    baseline.assert_called_once_with(config.data_path)
    v2.assert_not_called()
    prepare.assert_called_once_with(rows)
    assert prepared == prepare_records(rows)
    assert prepared.examples == [{"text": "normal", "label": 0}, {"text": "suspicious", "label": 1}]


def test_v2_selects_correct_columns_and_excludes_metadata():
    config = train.Config(data_format="v2", data_path="synthetic-v2.xlsx")
    rows = [
        {
            "content": text, "review": "wrong text column", "final_label": label,
            "use_for_training": flag, "source_set": "merged_rechecked", "source_id": 42,
            "source_note": "audit", "platform": "test", "rating": 5,
            "candidate_reasons": "reason", "confirmed_patterns": "pattern",
        }
        for text, label, flag in [
            ("normal", "NORMAL", " yes "),
            ("suspicious", "SUSPICIOUS", "YES"),
            ("no", "NORMAL", "NO"),
            ("uncertain", "UNCERTAIN", "YES"),
            ("invalid", "INVALID", "YES"),
            ("promo", "DISCLOSED_PROMO", "YES"),
        ]
    ]
    with (
        patch.object(train, "read_review_master") as baseline,
        patch.object(train, "read_ptext_v2_workbook", return_value=rows) as v2,
        patch.object(train, "prepare_records", wraps=prepare_records) as prepare,
    ):
        prepared = train.load_prepared_data(config)
    baseline.assert_not_called()
    v2.assert_called_once_with("synthetic-v2.xlsx")
    prepare.assert_called_once_with(
        rows, text_column="content", label_column="final_label",
        use_for_training_column="use_for_training",
    )
    assert prepared.examples == [{"text": "normal", "label": 0}, {"text": "suspicious", "label": 1}]
    assert all(set(example) == {"text", "label"} for example in prepared.examples)


def test_v2_cli_accepts_requested_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "train", "--data-format", "v2",
        "--data-path", "data/p_text_v2/ptext_v2_training_master.xlsx",
        "--output-dir", "artifacts/p_text_v2_alpha_01",
    ])
    config = train.parse_args()
    assert config.data_format == "v2"
    assert config.data_path == "data/p_text_v2/ptext_v2_training_master.xlsx"
    assert config.output_dir == "artifacts/p_text_v2_alpha_01"


def test_invalid_format_is_rejected_by_argparse(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["train", "--data-format", "unknown"])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_invalid_programmatic_format_does_not_read_workbooks():
    with patch.object(train, "read_review_master") as baseline, patch.object(train, "read_ptext_v2_workbook") as v2:
        with pytest.raises(ValueError, match="Unsupported data format"):
            train.load_prepared_data(train.Config(data_format="unknown"))
    baseline.assert_not_called()
    v2.assert_not_called()


@pytest.mark.parametrize("data_format", ["baseline", "v2"])
def test_training_config_serialization_includes_data_format(data_format):
    # main's existing training_config.json writer serializes asdict(config).
    record = json.loads(json.dumps(asdict(train.Config(data_format=data_format))))
    assert record["data_format"] == data_format
