"""CLI for the offline KoELECTRA P_text v0 baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .data import compute_class_weights, prepare_records, read_review_master, split_counts, stratified_split
from .metrics import classification_metrics


@dataclass
class Config:
    data_path: str = "data/ReView_Integrated_Review_Dataset_v3_2차검수.xlsx"
    output_dir: str = "artifacts/p_text_baseline"
    model_name: str = "monologg/koelectra-base-v3-discriminator"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 256
    seed: int = 42


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=Config.data_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--model-name", default=Config.model_name)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--max-length", type=int, default=Config.max_length)
    parser.add_argument("--seed", type=int, default=Config.seed)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    try:
        import numpy as np
        import torch
        from torch.nn import CrossEntropyLoss
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies. Install with: pip install -e '.[ml]'") from exc

    set_seed(config.seed)
    prepared = prepare_records(read_review_master(config.data_path))
    splits = stratified_split(prepared.examples, seed=config.seed)
    weights = compute_class_weights(splits["train"])
    output = Path(config.output_dir)
    model_dir = output / "model"
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    class TextDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            # Only text is passed to the tokenizer. Metadata is not accepted here.
            self.encodings = tokenizer(
                [row["text"] for row in rows], truncation=True, padding=False,
                max_length=config.max_length,
            )
            self.labels = [row["label"] for row in rows]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}

    # The checkpoint declares model_type="electra". AutoModel therefore selects
    # ElectraForSequenceClassification and initializes a new two-class head on
    # top of the pretrained discriminator parameters.
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = CrossEntropyLoss(weight=weight_tensor.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def trainer_metrics(prediction) -> dict[str, float]:
        predicted = np.argmax(prediction.predictions, axis=1).tolist()
        result = classification_metrics(prediction.label_ids.tolist(), predicted)
        return {key: value for key, value in result.items() if isinstance(value, float)} | {
            f"suspicious_{key}": value for key, value in result["suspicious"].items()
        }

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size, learning_rate=config.learning_rate,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="suspicious_f1", greater_is_better=True, seed=config.seed,
        data_seed=config.seed, report_to="none", save_total_limit=1,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=TextDataset(splits["train"]),
        eval_dataset=TextDataset(splits["validation"]), tokenizer=tokenizer,
        compute_metrics=trainer_metrics,
    )
    trainer.train()
    prediction = trainer.predict(TextDataset(splits["test"]))
    y_pred = np.argmax(prediction.predictions, axis=1).tolist()
    y_true = prediction.label_ids.tolist()
    metrics = classification_metrics(y_true, y_pred)
    metrics["class_weights"] = weights
    metrics["split_counts"] = split_counts(splits)
    metrics["data_summary"] = {
        "raw_selected": prepared.raw_selected_count,
        "used_after_deduplication": len(prepared.examples),
        "duplicate_rows_removed": prepared.duplicate_rows_removed,
        "conflicting_groups_removed": prepared.conflicting_groups_removed,
    }
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "confusion_matrix.json").write_text(json.dumps({"labels": ["NORMAL", "SUSPICIOUS"], "matrix": metrics["confusion_matrix"]}, indent=2), encoding="utf-8")
    config_record = asdict(config) | {"device": "cuda" if torch.cuda.is_available() else "cpu", "class_weights": weights, "split_counts": split_counts(splits)}
    (output / "training_config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
