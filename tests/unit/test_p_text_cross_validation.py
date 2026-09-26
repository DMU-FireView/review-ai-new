import json
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.training.p_text import cross_validate_v2 as cv, train
from app.training.p_text.data import InsufficientDataError, prepare_records


@pytest.fixture
def real():
    return prepare_records([
        {"content": f"Real {label} {i}", "final_label": label, "use_for_training": "YES", "note": "audit"}
        for label, count in [("NORMAL", 30), ("SUSPICIOUS", 15)] for i in range(count)
    ], text_column="content", use_for_training_column="use_for_training")


def synthetic(text, label="SUSPICIOUS", flag="YES"):
    return {
        "content": text, "final_label": label, "use_for_training": flag,
        "synthetic_id": "s1", "source_set": "synthetic", "risk_type": "test",
        "split_policy": "TRAIN_ONLY", "generation_note": "metadata",
    }


def test_cli_defaults_and_baseline_unchanged():
    args = cv.parse_args([])
    assert args.folds == 3 and args.seed == 42 and args.epochs == 1 and args.threshold == .5
    assert args.data_format == "v2" and args.train_augmentation_path is None
    assert args.data_path == "data/p_text_v2/ptext_v2_training_master.xlsx"
    for name in ("model_name", "batch_size", "max_length", "learning_rate"):
        assert getattr(args, name) == getattr(train.Config(), name)
    assert train.Config().epochs == 3 and train.Config().data_format == "baseline"
    supplied = cv.parse_args(["--folds", "5", "--train-augmentation-path", "synthetic.xlsx", "--output-dir", "new-output"])
    assert supplied.folds == 5 and supplied.train_augmentation_path == "synthetic.xlsx"


@pytest.mark.parametrize("arguments", [["--folds", "1"], ["--threshold", "nan"], ["--threshold", "1.1"], ["--epochs", "0"]])
def test_invalid_cli(arguments):
    with pytest.raises(SystemExit):
        cv.parse_args(arguments)


def test_stratification_exactly_once_reproducible_and_balanced(real):
    folds = cv.stratified_folds(real.examples, 3, 42)
    assert folds == cv.stratified_folds(real.examples, 3, 42)
    assert folds != cv.stratified_folds(real.examples, 3, 43)
    assert sorted(i for _, evaluation in folds for i in evaluation) == list(range(45))
    for training, evaluation in folds:
        assert not set(training) & set(evaluation)
        assert sorted(training + evaluation) == list(range(45))
        assert sum(real.examples[i]["label"] for i in evaluation) == 5
    uneven = cv.stratified_folds(real.examples[:-1], 3, 42)
    assert max(len(e) for _, e in uneven) - min(len(e) for _, e in uneven) <= 1
    positives = [sum(real.examples[i]["label"] for i in e) for _, e in uneven]
    assert max(positives) - min(positives) <= 1


def test_insufficient_real_class_cannot_be_fixed_with_synthetic():
    with pytest.raises(InsufficientDataError):
        cv.stratified_folds([{"text": str(i), "label": i % 2} for i in range(4)], 3, 42)


@pytest.mark.parametrize("augmentation", [False, True])
def test_cv_orchestration_train_only_fixed_threshold_and_output(real, tmp_path, augmentation):
    output = tmp_path / "cv"
    config = cv.Config(output_dir=str(output), train_augmentation_path="synthetic.xlsx" if augmentation else None)
    original = [dict(row) for row in real.examples]
    rows = [synthetic("New synthetic"), synthetic(" NEW\nSYNTHETIC ")]
    rows += [synthetic("  " + row["text"].upper().replace(" ", "\t") + " ") for row in real.examples]
    partitions = cv.stratified_folds(real.examples, 3, 42)
    calls = []

    def fake_fold(config, training, evaluation, weights, directory):
        index = len(calls)
        train_indices, evaluation_indices = partitions[index]
        assert evaluation == [real.examples[i] for i in evaluation_indices]
        expected = [real.examples[i] for i in train_indices]
        assert training == expected + ([{"text": "New synthetic", "label": 1}] if augmentation else [])
        assert all(set(row) == {"text", "label"} for row in training + evaluation)
        assert weights == pytest.approx([31 / 40, 31 / 22] if augmentation else [.75, 1.5])
        calls.append(directory)
        # All predictions are suspicious at >=0.5, exposing any argmax/tuning drift.
        return [.5] * len(evaluation)

    with (
        patch.object(cv, "load_prepared_data", return_value=real) as loader,
        patch.object(train, "read_ptext_v2_workbook", return_value=rows) as synthetic_loader,
        patch.object(cv, "train_fold", side_effect=fake_fold),
        patch("app.training.p_text.diagnose_threshold.select_threshold", side_effect=AssertionError("No tuning")),
    ):
        result = cv.run_cross_validation(config)
    loader.assert_called_once_with(config)
    assert synthetic_loader.call_count == (3 if augmentation else 0)
    assert len(calls) == 3 and real.examples == original
    assert result["pooled_metrics"]["confusion_matrix"] == [[0, 30], [0, 15]]
    assert result["fold_metric_mean"]["suspicious_f1"] == .5
    assert result["fold_metric_std"]["accuracy"] == 0
    for number, fold in enumerate(result["fold_results"], 1):
        assert fold["threshold"] == .5
        assert fold["real_split_counts"]["evaluation"] == {"NORMAL": 10, "SUSPICIOUS": 5}
        assert fold["final_split_counts"]["evaluation"] == fold["real_split_counts"]["evaluation"]
        assert fold["augmentation_added_to_train"] == int(augmentation)
        assert fold["augmentation_raw_selected"] == (47 if augmentation else 0)
        assert fold["augmentation_used_after_deduplication"] == (46 if augmentation else 0)
        assert fold["augmentation_real_collision_removed"] == (45 if augmentation else 0)
        assert json.loads((output / f"fold_{number}" / "metrics.json").read_text(encoding="utf-8")) == fold
    stored = json.loads((output / "cross_validation_metrics.json").read_text(encoding="utf-8"))
    assert stored == result
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["real_index"] for row in predictions] == list(range(45))
    assert all(row["predicted_label"] == 1 and row["probability_suspicious"] == .5 for row in predictions)
    assert sum(row["actual_label"] for row in predictions) == 15


@pytest.mark.parametrize("rows", [
    [], [synthetic(None)], [synthetic("ignored", flag="NO")],
    [synthetic("normal", label="NORMAL")],
    [synthetic("same"), synthetic(" SAME ", label="NORMAL")],
])
def test_invalid_synthetic_stops_before_model_or_output(real, tmp_path, rows):
    output = tmp_path / "cv"
    with (
        patch.object(cv, "load_prepared_data", return_value=real),
        patch.object(train, "read_ptext_v2_workbook", return_value=rows),
        patch.object(cv, "train_fold") as model,
        pytest.raises(ValueError),
    ):
        cv.run_cross_validation(cv.Config(output_dir=str(output), train_augmentation_path="synthetic.xlsx"))
    model.assert_not_called()
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    with patch.object(cv, "load_prepared_data") as loader, pytest.raises(FileExistsError):
        cv.run_cross_validation(cv.Config(output_dir=str(tmp_path)))
    loader.assert_not_called()


def test_aggregate_mean_population_std_and_pooled_metrics():
    perfect = cv.evaluate_predictions([0, 1], [.1, .9], .5)
    wrong = cv.evaluate_predictions([0, 1], [.9, .1], .5)
    predictions = [
        {"real_index": i, "actual_label": label, "probability_suspicious": probability}
        for i, (label, probability) in enumerate([(0, .1), (1, .9), (0, .9), (1, .1)])
    ]
    results = [{"metrics": perfect}, {"metrics": wrong}]
    summary = cv.aggregate_metrics(results, predictions, 4, .5)
    assert all(value == .5 for value in summary["fold_metric_mean"].values())
    assert all(value == .5 for value in summary["fold_metric_std"].values())
    assert summary["pooled_metrics"]["confusion_matrix"] == [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="exactly one"):
        cv.aggregate_metrics(results, predictions + [predictions[0]], 4, .5)


def test_fold_backend_fresh_models_seed_loss_and_no_evaluation_selection(tmp_path):
    # Entire ML backend is mocked: no real torch/transformers/model is loaded.
    transformers = ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = MagicMock()
    models = [MagicMock(), MagicMock(), MagicMock()]
    transformers.AutoModelForSequenceClassification.from_pretrained.side_effect = models
    transformers.AutoTokenizer = MagicMock()
    tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
    tokenizer.side_effect = lambda texts, **kwargs: {"input_ids": [[1] for _ in texts]}
    transformers.TrainingArguments = MagicMock()
    transformers.set_seed = MagicMock()
    instances = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.events = []
            instances.append(self)

        def train(self):
            self.events.append("train")

        def predict(self, dataset):
            assert self.events == ["train"]
            assert len(dataset) == 2 and dataset[0] == {"input_ids": [1], "labels": 0}
            self.events.append("predict")
            return SimpleNamespace(predictions=np.array([[0., 0.], [0., 2.]]))

        def save_model(self, path):
            self.events.append("save")

    transformers.Trainer = FakeTrainer
    torch = ModuleType("torch")
    torch.tensor = MagicMock()
    torch.float = "float"
    nn = ModuleType("torch.nn")
    nn.CrossEntropyLoss = MagicMock()
    dataset_module = ModuleType("torch.utils.data")
    dataset_module.Dataset = object
    modules = {"transformers": transformers, "torch": torch, "torch.nn": nn, "torch.utils.data": dataset_module}
    rows = [{"text": "normal", "label": 0}, {"text": "suspicious", "label": 1}]
    with patch.dict("sys.modules", modules):
        for fold in range(3):
            probabilities = cv.train_fold(cv.Config(), rows, rows, [1., 1.], tmp_path / str(fold))
            assert probabilities == pytest.approx([.5, .8807970779])
    assert [instance.kwargs["model"] for instance in instances] == models
    assert all("eval_dataset" not in instance.kwargs for instance in instances)
    assert all(instance.events == ["train", "predict", "save"] for instance in instances)
    assert transformers.set_seed.call_count == 3
    for call in transformers.set_seed.call_args_list:
        assert call.args == (42,)
    for call in transformers.AutoModelForSequenceClassification.from_pretrained.call_args_list:
        assert call.args == (train.Config.model_name,) and call.kwargs == {"num_labels": 2}
    for call in transformers.TrainingArguments.call_args_list:
        assert call.kwargs["eval_strategy"] == "no" and call.kwargs["save_strategy"] == "no"
        assert call.kwargs["load_best_model_at_end"] is False
        assert call.kwargs["seed"] == call.kwargs["data_seed"] == 42
    for call in tokenizer.call_args_list:
        assert call.args == (["normal", "suspicious"],)
    instances[0].compute_loss(models[0], {"labels": "labels", "input_ids": "ids"})
    nn.CrossEntropyLoss.assert_called_once_with(weight=torch.tensor.return_value.to.return_value)
