#!/usr/bin/env python3
"""Fine-tune a shared DeBERTa encoder with heads for all power tasks."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as exc:
    raise SystemExit(
        "Missing transformer dependencies. Install them with:\n"
        "pip install torch transformers scikit-learn numpy"
    ) from exc

try:
    from modeling.data import RATING_LABELS, STRATEGIES
except ModuleNotFoundError:
    from data import RATING_LABELS, STRATEGIES


RATING_TO_ID = {label: i for i, label in enumerate(RATING_LABELS)}
ID_TO_RATING = {i: label for label, i in RATING_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_text(example: dict) -> str:
    return (
        f"Character A: {example['character_a']}\n"
        f"Character B: {example['character_b']}\n\n"
        f"{example['text']}"
    )


class PowerDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            format_text(ex),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        strategies = [1.0 if strategy in ex["power_strategies"] else 0.0 for strategy in STRATEGIES]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "rating": torch.tensor(RATING_TO_ID[ex["power_rating"]], dtype=torch.long),
            "shift": torch.tensor(int(ex["power_shift"]), dtype=torch.long),
            "strategies": torch.tensor(strategies, dtype=torch.float),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class MultiTaskDeberta(nn.Module):
    def __init__(self, model_name: str, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.rating_head = nn.Linear(hidden_size, len(RATING_LABELS))
        self.shift_head = nn.Linear(hidden_size, 2)
        self.strategy_head = nn.Linear(hidden_size, len(STRATEGIES))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        pooled = pooled.to(self.rating_head.weight.dtype)
        return {
            "rating_logits": self.rating_head(pooled),
            "shift_logits": self.shift_head(pooled),
            "strategy_logits": self.strategy_head(pooled),
        }


def split_examples(examples: list[dict], test_size: float, seed: int) -> tuple[list[int], list[int]]:
    ratings = np.array([ex["power_rating"] for ex in examples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(ratings)), ratings))


def class_weights(labels: list[int], n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 5.0)
    return torch.tensor(weights, dtype=torch.float, device=device)


def strategy_pos_weight(examples: list[dict], device) -> torch.Tensor:
    y = np.array(
        [[1 if strategy in ex["power_strategies"] else 0 for strategy in STRATEGIES] for ex in examples],
        dtype=np.float32,
    )
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float, device=device)


def batch_to_device(batch: dict, device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, device) -> tuple[dict, dict]:
    model.eval()
    rating_true, rating_pred = [], []
    shift_true, shift_pred = [], []
    strategy_true, strategy_pred = [], []
    rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            rating_probs = torch.softmax(outputs["rating_logits"], dim=-1)
            shift_probs = torch.softmax(outputs["shift_logits"], dim=-1)
            strategy_probs = torch.sigmoid(outputs["strategy_logits"])

            batch_rating_pred = rating_probs.argmax(dim=-1)
            batch_shift_pred = shift_probs.argmax(dim=-1)
            batch_strategy_pred = (strategy_probs >= 0.5).long()

            rating_true.extend(batch["rating"].cpu().tolist())
            rating_pred.extend(batch_rating_pred.cpu().tolist())
            shift_true.extend(batch["shift"].cpu().tolist())
            shift_pred.extend(batch_shift_pred.cpu().tolist())
            strategy_true.extend(batch["strategies"].cpu().numpy().astype(int).tolist())
            strategy_pred.extend(batch_strategy_pred.cpu().numpy().astype(int).tolist())

            for i in range(len(batch["index"])):
                rows.append(
                    {
                        "index": int(batch["index"][i].cpu()),
                        "rating_probs": rating_probs[i].cpu().tolist(),
                        "shift_probs": shift_probs[i].cpu().tolist(),
                        "strategy_probs": strategy_probs[i].cpu().tolist(),
                        "rating_pred": int(batch_rating_pred[i].cpu()),
                        "shift_pred": int(batch_shift_pred[i].cpu()),
                        "strategy_pred": batch_strategy_pred[i].cpu().tolist(),
                    }
                )

    metrics = {
        "rating_accuracy": accuracy_score(rating_true, rating_pred),
        "rating_macro_f1": f1_score(rating_true, rating_pred, average="macro", zero_division=0),
        "shift_accuracy": accuracy_score(shift_true, shift_pred),
        "shift_macro_f1": f1_score(shift_true, shift_pred, average="macro", zero_division=0),
        "strategy_micro_f1": f1_score(strategy_true, strategy_pred, average="micro", zero_division=0),
        "strategy_macro_f1": f1_score(strategy_true, strategy_pred, average="macro", zero_division=0),
        "strategy_hamming_loss": hamming_loss(strategy_true, strategy_pred),
        "rating_report": classification_report(
            rating_true, rating_pred, labels=list(range(len(RATING_LABELS))), target_names=RATING_LABELS, zero_division=0
        ),
        "shift_report": classification_report(
            shift_true, shift_pred, labels=[0, 1], zero_division=0
        ),
        "strategy_report": classification_report(
            strategy_true, strategy_pred, target_names=STRATEGIES, zero_division=0
        ),
    }
    metrics["selection_score"] = float(
        np.mean([metrics["rating_macro_f1"], metrics["shift_macro_f1"], metrics["strategy_micro_f1"]])
    )
    return metrics, {row["index"]: row for row in rows}


def save_prediction_csv(
    output_path: Path,
    model_name: str,
    examples: list[dict],
    split_names: list[str],
    prediction_rows: dict[int, dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "split",
                "doc_id",
                "episode",
                "character_a",
                "character_b",
                "true_power_rating",
                "pred_power_rating",
                "power_rating_probs",
                "true_power_shift",
                "pred_power_shift",
                "power_shift_probs",
                "true_power_strategies",
                "pred_power_strategies",
                "power_strategy_probs",
                "text",
            ],
        )
        writer.writeheader()
        for idx, ex in enumerate(examples):
            pred = prediction_rows[idx]
            pred_strategies = [
                strategy for strategy, value in zip(STRATEGIES, pred["strategy_pred"]) if int(value) == 1
            ]
            writer.writerow(
                {
                    "model_name": model_name,
                    "split": split_names[idx],
                    "doc_id": ex["doc_id"],
                    "episode": ex["episode"],
                    "character_a": ex["character_a"],
                    "character_b": ex["character_b"],
                    "true_power_rating": ex["power_rating"],
                    "pred_power_rating": ID_TO_RATING[pred["rating_pred"]],
                    "power_rating_probs": json.dumps(
                        {label: float(prob) for label, prob in zip(RATING_LABELS, pred["rating_probs"])}
                    ),
                    "true_power_shift": int(ex["power_shift"]),
                    "pred_power_shift": int(pred["shift_pred"]),
                    "power_shift_probs": json.dumps(
                        {str(label): float(prob) for label, prob in zip([0, 1], pred["shift_probs"])}
                    ),
                    "true_power_strategies": json.dumps(ex["power_strategies"]),
                    "pred_power_strategies": json.dumps(pred_strategies),
                    "power_strategy_probs": json.dumps(
                        {label: float(prob) for label, prob in zip(STRATEGIES, pred["strategy_probs"])}
                    ),
                    "text": ex["text"],
                }
            )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_score: float,
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score,
            "history": history,
            "args": vars(args),
        },
        path,
    )


def load_model_weights(path: Path, model, device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="modeling/dataset.json")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--run-name", default="deberta_v3_base_multitask")
    parser.add_argument("--output-dir", default="modeling/artifacts/deberta")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--rating-loss-weight", type=float, default=1.0)
    parser.add_argument("--shift-loss-weight", type=float, default=0.7)
    parser.add_argument("--strategy-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a full checkpoint saved by this script, usually last_checkpoint.pt.",
    )
    parser.add_argument(
        "--init-from-model",
        default=None,
        help="Path to model weights, such as best_model.pt. Optimizer state starts fresh.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    examples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    train_idx, test_idx = split_examples(examples, args.test_size, args.seed)
    train_examples = [examples[i] for i in train_idx]
    test_examples = [examples[i] for i in test_idx]
    split_names = ["train"] * len(examples)
    for i in test_idx:
        split_names[i] = "test"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        PowerDataset(train_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        PowerDataset(test_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    all_loader = DataLoader(
        PowerDataset(examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MultiTaskDeberta(args.model_name, args.dropout).to(device)
    model.float()
    rating_loss = nn.CrossEntropyLoss(
        weight=class_weights([RATING_TO_ID[ex["power_rating"]] for ex in train_examples], len(RATING_LABELS), device)
    )
    shift_loss = nn.CrossEntropyLoss(
        weight=class_weights([int(ex["power_shift"]) for ex in train_examples], 2, device)
    )
    strategy_loss = nn.BCEWithLogitsLoss(pos_weight=strategy_pos_weight(train_examples, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []
    start_epoch = 1

    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_score = float(checkpoint.get("best_score", -1.0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming full checkpoint from epoch {checkpoint['epoch']}: {args.resume_from_checkpoint}")
    elif args.init_from_model:
        load_model_weights(Path(args.init_from_model), model, device)
        print(f"Initialized model weights from {args.init_from_model}; optimizer starts fresh.")

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch - 1}, but --epochs is {args.epochs}. "
            "Set --epochs to the final total epoch count you want, not the number of extra epochs."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        rating_losses = []
        shift_losses = []
        strategy_losses = []
        skipped_batches = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            rating_component = rating_loss(outputs["rating_logits"].float(), batch["rating"])
            shift_component = shift_loss(outputs["shift_logits"].float(), batch["shift"])
            strategy_component = strategy_loss(outputs["strategy_logits"].float(), batch["strategies"])
            loss = (
                args.rating_loss_weight * rating_component
                + args.shift_loss_weight * shift_component
                + args.strategy_loss_weight * strategy_component
            )
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            rating_losses.append(float(rating_component.detach().cpu()))
            shift_losses.append(float(shift_component.detach().cpu()))
            strategy_losses.append(float(strategy_component.detach().cpu()))

        if not losses:
            raise RuntimeError(
                "Every training batch produced a non-finite loss. Try a lower learning rate, "
                "check that the runtime is not forcing float16, or switch to microsoft/deberta-v3-small."
            )

        metrics, _ = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        metrics["rating_train_loss"] = float(np.mean(rating_losses))
        metrics["shift_train_loss"] = float(np.mean(shift_losses))
        metrics["strategy_train_loss"] = float(np.mean(strategy_losses))
        metrics["skipped_batches"] = skipped_batches
        history.append(metrics)
        print(
            f"epoch={epoch} loss={metrics['train_loss']:.4f} "
            f"rating_loss={metrics['rating_train_loss']:.4f} "
            f"shift_loss={metrics['shift_train_loss']:.4f} "
            f"strategy_loss={metrics['strategy_train_loss']:.4f} "
            f"skipped={skipped_batches} "
            f"rating_macro_f1={metrics['rating_macro_f1']:.3f} "
            f"shift_macro_f1={metrics['shift_macro_f1']:.3f} "
            f"strategy_micro_f1={metrics['strategy_micro_f1']:.3f}"
        )

        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        torch.save(model.state_dict(), output_dir / "latest_model.pt")
        save_checkpoint(
            output_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            history,
            args,
        )

    load_model_weights(output_dir / "best_model.pt", model, device)
    test_metrics, _ = evaluate(model, test_loader, device)
    _, all_predictions = evaluate(model, all_loader, device)

    tokenizer.save_pretrained(output_dir / "tokenizer")
    config = vars(args) | {"rating_labels": RATING_LABELS, "strategy_labels": STRATEGIES}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "best_test_metrics": test_metrics}, indent=2),
        encoding="utf-8",
    )
    save_prediction_csv(
        output_dir / "predictions.csv",
        args.run_name,
        examples,
        split_names,
        all_predictions,
    )

    print("\nBest test metrics")
    print(test_metrics["rating_report"])
    print(test_metrics["shift_report"])
    print(test_metrics["strategy_report"])
    print(f"Saved DeBERTa artifacts to {output_dir}")


if __name__ == "__main__":
    main()
