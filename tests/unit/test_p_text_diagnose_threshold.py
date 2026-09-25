from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from app.training.p_text import diagnose_threshold as diagnostic
from app.training.p_text.data import prepare_records


def test_cli_defaults_preserve_baseline(monkeypatch):
    monkeypatch.setattr("sys.argv", ["diagnose_threshold"])
    args = diagnostic.parse_args()
    assert args.data_format == "baseline"
    assert args.data_path == "data/ReView_Integrated_Review_Dataset_v4_3_전체재검수_SUSPICIOUS확정.xlsx"
    assert args.model_path == "artifacts/p_text_baseline_v4_3/model"
    assert (args.seed, args.batch_size, args.max_length) == (42, 8, 256)


def test_baseline_uses_original_loader_and_preparation():
    rows = [
        {"review": "normal", "final_label": "NORMAL", "use_for_training": "NO"},
        {"review": "suspicious", "final_label": "SUSPICIOUS", "platform": "audit"},
    ]
    with (
        patch.object(diagnostic, "read_review_master", return_value=rows) as baseline,
        patch.object(diagnostic, "read_ptext_v2_workbook") as v2,
        patch.object(diagnostic, "prepare_records", wraps=prepare_records) as prepare,
    ):
        prepared = diagnostic.load_prepared_data(Namespace(data_format="baseline", data_path="synthetic.xlsx"))
    baseline.assert_called_once_with("synthetic.xlsx")
    v2.assert_not_called()
    prepare.assert_called_once_with(rows)
    assert prepared.examples == [{"text": "normal", "label": 0}, {"text": "suspicious", "label": 1}]


def test_v2_uses_correct_loader_columns_and_text_only_examples():
    rows = [
        {
            "content": text, "review": "wrong column", "final_label": label,
            "use_for_training": flag, "source_set": "merged_rechecked", "source_id": 10,
            "source_note": "audit", "platform": "audit", "rating": 5,
            "candidate_reasons": "reason", "confirmed_patterns": "pattern",
        }
        for text, label, flag in [
            ("normal", "NORMAL", " yes "),
            ("suspicious", "SUSPICIOUS", "YES"),
            ("excluded", "NORMAL", "NO"),
            ("uncertain", "UNCERTAIN", "YES"),
            ("invalid", "INVALID", "YES"),
            ("promo", "DISCLOSED_PROMO", "YES"),
        ]
    ]
    with (
        patch.object(diagnostic, "read_review_master") as baseline,
        patch.object(diagnostic, "read_ptext_v2_workbook", return_value=rows) as v2,
        patch.object(diagnostic, "prepare_records", wraps=prepare_records) as prepare,
    ):
        prepared = diagnostic.load_prepared_data(Namespace(data_format="v2", data_path="synthetic.xlsx"))
    baseline.assert_not_called()
    v2.assert_called_once_with("synthetic.xlsx")
    prepare.assert_called_once_with(
        rows, text_column="content", label_column="final_label",
        use_for_training_column="use_for_training",
    )
    assert prepared.examples == [{"text": "normal", "label": 0}, {"text": "suspicious", "label": 1}]
    assert all(set(row) == {"text", "label"} for row in prepared.examples)


def test_v2_cli_accepts_requested_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "diagnose_threshold", "--data-format", "v2",
        "--data-path", "data/p_text_v2/ptext_v2_training_master.xlsx",
        "--model-path", "artifacts/p_text_v2_smoke_01/model",
    ])
    args = diagnostic.parse_args()
    assert args.data_format == "v2"
    assert args.data_path == "data/p_text_v2/ptext_v2_training_master.xlsx"
    assert args.model_path == "artifacts/p_text_v2_smoke_01/model"


def test_invalid_format_is_rejected_by_argparse(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["diagnose_threshold", "--data-format", "invalid"])
    with pytest.raises(SystemExit) as error:
        diagnostic.parse_args()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_threshold_range_is_unchanged():
    assert diagnostic.THRESHOLDS == (
        .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
        .55, .60, .65, .70, .75, .80, .85, .90, .95,
    )


@pytest.mark.parametrize("scores,expected", [
    ({.20: (.9, .5), .50: (.8, 1)}, .20),  # F1 outranks recall and proximity.
    ({.20: (.8, 1), .50: (.8, .5)}, .20),  # Recall breaks an F1 tie.
    ({.20: (.8, 1), .50: (.8, 1)}, .50),  # Proximity breaks both ties.
])
def test_threshold_selection_priorities(scores, expected):
    def fake_metrics(labels, probabilities, threshold):
        f1, recall = scores.get(threshold, (0, 0))
        return {"suspicious": {"f1": f1, "recall": recall}}

    with patch.object(diagnostic, "metrics_at_threshold", side_effect=fake_metrics) as metrics:
        threshold, result = diagnostic.select_threshold([0, 1], [.1, .9])
    assert threshold == expected
    assert (result["suspicious"]["f1"], result["suspicious"]["recall"]) == scores[expected]
    assert [call.args[2] for call in metrics.call_args_list] == list(diagnostic.THRESHOLDS)


def test_real_metrics_and_threshold_boundary():
    metrics = diagnostic.metrics_at_threshold([0, 0, 1, 1], [.1, .5, .5, .9], .5)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == .75
    assert metrics["suspicious"] == {"precision": 2 / 3, "recall": 1.0, "f1": .8}
    threshold, selected = diagnostic.select_threshold([0, 1], [.1, .9])
    assert threshold == .5
    assert selected["suspicious"]["f1"] == 1.0


def test_prediction_tokenizes_only_text_without_real_model():
    tokenizer = MagicMock()
    tensor = MagicMock()
    tokenizer.return_value = {"input_ids": tensor}
    model = MagicMock()
    torch = MagicMock()
    torch.softmax.return_value.__getitem__.return_value.cpu.return_value.tolist.return_value = [.7]
    probabilities = diagnostic.predict_probabilities(
        [{"text": "review only", "label": 1, "source_note": "metadata"}],
        model=model, tokenizer=tokenizer, torch=torch, device="cpu", batch_size=8, max_length=256,
    )
    assert probabilities == [.7]
    tokenizer.assert_called_once_with(
        ["review only"], truncation=True, padding=True, max_length=256, return_tensors="pt",
    )
    model.assert_called_once_with(input_ids=tensor.to.return_value)
