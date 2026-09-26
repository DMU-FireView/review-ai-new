"""Real-only stratified CV with optional synthetic train augmentation.

No evaluation-based checkpoint or threshold selection. Precision/recall/F1 are
existing macro metrics; std is population std (ddof=0). Output must be new.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Any

from .data import InsufficientDataError, compute_class_weights
from .metrics import classification_metrics
from .train import Config as TrainingConfig, augment_training_splits, load_prepared_data


@dataclass
class Config(TrainingConfig):
    data_path: str = "data/p_text_v2/ptext_v2_training_master.xlsx"
    data_format: str = "v2"
    output_dir: str = "artifacts/p_text_v2_cv_01"
    epochs: int = 1
    folds: int = 3
    threshold: float = 0.5


def validate_config(config: Config) -> None:
    if config.data_format != "v2":
        raise ValueError("Cross-validation requires v2 data")
    if config.folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 <= config.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if min(config.epochs, config.batch_size, config.max_length) < 1:
        raise ValueError("epochs, batch-size and max-length must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_path", "train_augmentation_path", "output_dir", "model_name"):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(Config, name))
    for name in ("folds", "seed", "epochs", "batch_size", "max_length"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Config, name))
    for name in ("threshold", "learning_rate"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Config, name))
    config = Config(**vars(parser.parse_args(argv)))
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def stratified_folds(examples: list[dict[str, Any]], folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Seeded class-wise shuffle; class and total evaluation sizes differ by <=1.

    Dependency-free stratified K-fold, not sklearn's exact index ordering.
    Returned train/evaluation subsets retain real input order.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    classes: dict[int, list[int]] = {0: [], 1: []}
    for index, row in enumerate(examples):
        if row["label"] not in classes:
            raise ValueError("Only binary labels are supported")
        classes[row["label"]].append(index)
    if any(len(indices) < folds for indices in classes.values()):
        raise InsufficientDataError("Each real class needs at least folds unique examples")
    rng = random.Random(seed)
    assignments = [set() for _ in range(folds)]
    offset = 0
    for indices in classes.values():
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignments[(offset + position) % folds].add(index)
        offset = (offset + len(indices)) % folds
    return [
        ([i for i in range(len(examples)) if i not in held_out], sorted(held_out))
        for held_out in assignments
    ]


def evaluate_predictions(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    metrics = classification_metrics(labels, [int(p >= threshold) for p in probabilities])
    return metrics | {f"suspicious_{name}": value for name, value in metrics["suspicious"].items()}


def aggregate_metrics(fold_results: list[dict[str, Any]], predictions: list[dict[str, Any]], real_count: int, threshold: float) -> dict[str, Any]:
    if sorted(row["real_index"] for row in predictions) != list(range(real_count)):
        raise ValueError("Every real example must have exactly one evaluation prediction")
    names = ("accuracy", "precision", "recall", "f1", "suspicious_precision", "suspicious_recall", "suspicious_f1")
    return {
        "fold_metric_mean": {name: mean(fold["metrics"][name] for fold in fold_results) for name in names},
        "fold_metric_std": {name: pstdev(fold["metrics"][name] for fold in fold_results) for name in names},
        "pooled_metrics": evaluate_predictions(
            [row["actual_label"] for row in predictions],
            [row["probability_suspicious"] for row in predictions], threshold,
        ),
    }


def train_fold(config: Config, rows: list[dict[str, Any]], evaluation: list[dict[str, Any]], weights: list[float], output: Path) -> list[float]:
    """Fresh checkpoint and optimizer each call; evaluate after all epochs."""
    import numpy as np
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)

    class TextDataset(Dataset):
        def __init__(self, examples):
            self.encodings = tokenizer(
                [row["text"] for row in examples], truncation=True, padding=False,
                max_length=config.max_length,
            )
            self.labels = [row["label"] for row in examples]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}

    weight_tensor = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = CrossEntropyLoss(weight=weight_tensor.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate, eval_strategy="no", save_strategy="no",
        load_best_model_at_end=False, seed=config.seed, data_seed=config.seed, report_to="none",
    )
    trainer = WeightedTrainer(model=model, args=arguments, train_dataset=TextDataset(rows), tokenizer=tokenizer)
    trainer.train()
    logits = np.asarray(trainer.predict(TextDataset(evaluation)).predictions)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = (exp[:, 1] / exp.sum(axis=1)).tolist()
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(str(output / "model"))
    return probabilities


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run_cross_validation(config: Config) -> dict[str, Any]:
    validate_config(config)
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    prepared = load_prepared_data(config)
    real = prepared.examples
    indices = stratified_folds(real, config.folds, config.seed)
    # Validate every fold before creating outputs or loading models.
    plans = []
    for train_indices, evaluation_indices in indices:
        splits, audit = augment_training_splits(config, {
            "train": [dict(real[i]) for i in train_indices],
            "evaluation": [dict(real[i]) for i in evaluation_indices],
        })
        plans.append((splits, audit, compute_class_weights(splits["train"]), evaluation_indices))
    output.mkdir(parents=True, exist_ok=False)
    fold_results, predictions = [], []
    for number, (splits, audit, weights, evaluation_indices) in enumerate(plans, start=1):
        fold_output = output / f"fold_{number}"
        fold_output.mkdir()
        probabilities = train_fold(config, splits["train"], splits["evaluation"], weights, fold_output)
        metrics = evaluate_predictions([row["label"] for row in splits["evaluation"]], probabilities, config.threshold)
        result = {"fold": number, "threshold": config.threshold, "class_weights": weights, "metrics": metrics, **audit}
        fold_results.append(result)
        write_json(fold_output / "metrics.json", result)
        for index, probability in zip(evaluation_indices, probabilities, strict=True):
            predictions.append({
                "fold": number, "real_index": index, "text": real[index]["text"],
                "actual_label": real[index]["label"], "probability_suspicious": probability,
                "predicted_label": int(probability >= config.threshold),
            })
    summary = {
        **asdict(config), "augmentation_enabled": config.train_augmentation_path is not None,
        "augmentation_path": config.train_augmentation_path,
        "real_data_summary": {
            "raw_selected": prepared.raw_selected_count, "used_after_deduplication": len(real),
            "duplicate_rows_removed": prepared.duplicate_rows_removed,
            "conflicting_groups_removed": prepared.conflicting_groups_removed,
            "NORMAL": sum(row["label"] == 0 for row in real),
            "SUSPICIOUS": sum(row["label"] == 1 for row in real),
        },
        "std_ddof": 0, "fold_results": fold_results,
        **aggregate_metrics(fold_results, predictions, len(real), config.threshold),
    }
    write_json(output / "cross_validation_metrics.json", summary)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in sorted(predictions, key=lambda row: row["real_index"]):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main() -> None:
    summary = run_cross_validation(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
