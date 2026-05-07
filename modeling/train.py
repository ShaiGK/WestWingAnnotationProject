#!/usr/bin/env python3
"""Train and evaluate baseline multi-task power dynamics models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from modeling.data import RATING_LABELS, STRATEGIES, build_dataset
except ModuleNotFoundError:
    from data import RATING_LABELS, STRATEGIES, build_dataset


def make_text_classifier(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_strategy_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def load_or_build_examples(args: argparse.Namespace) -> list[dict]:
    dataset_path = Path(args.dataset)
    if dataset_path.exists() and not args.rebuild_dataset:
        return json.loads(dataset_path.read_text(encoding="utf-8"))
    return build_dataset(
        annotations_path=Path(args.annotations),
        dialogues_dir=Path(args.dialogues),
        output_path=dataset_path,
        skip_missing_dialogues=args.skip_missing_dialogues,
    )


def train_test_indices(ratings: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(ratings)), ratings))


def evaluate_single_task(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "classification_report": report,
    }
    print(f"\n{name}")
    print(report)
    print(f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
    return metrics


def _single_task_probabilities(model: Pipeline, texts: np.ndarray) -> list[dict[str, float] | None]:
    if not hasattr(model, "predict_proba"):
        return [None for _ in texts]
    probs = model.predict_proba(texts)
    classes = [str(label) for label in model.classes_]
    return [
        {label: float(prob) for label, prob in zip(classes, row)}
        for row in probs
    ]


def _multi_label_probabilities(model: Pipeline, texts: np.ndarray, labels: list[str]) -> list[dict[str, float] | None]:
    if not hasattr(model, "predict_proba"):
        return [None for _ in texts]
    probs = model.predict_proba(texts)
    return [
        {label: float(prob) for label, prob in zip(labels, row)}
        for row in probs
    ]


def save_predictions(
    output_path: Path,
    model_name: str,
    examples: list[dict],
    split_names: np.ndarray,
    texts: np.ndarray,
    rating_model: Pipeline,
    shift_model: Pipeline,
    strategy_model: Pipeline,
    y_rating: np.ndarray,
    y_shift: np.ndarray,
    y_strategies: np.ndarray,
) -> None:
    rating_pred = rating_model.predict(texts)
    shift_pred = shift_model.predict(texts)
    strategy_pred = strategy_model.predict(texts)
    rating_probs = _single_task_probabilities(rating_model, texts)
    shift_probs = _single_task_probabilities(shift_model, texts)
    strategy_probs = _multi_label_probabilities(strategy_model, texts, STRATEGIES)

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
        for i, ex in enumerate(examples):
            true_strategies = [
                strategy for strategy, value in zip(STRATEGIES, y_strategies[i]) if int(value) == 1
            ]
            pred_strategies = [
                strategy for strategy, value in zip(STRATEGIES, strategy_pred[i]) if int(value) == 1
            ]
            writer.writerow(
                {
                    "model_name": model_name,
                    "split": split_names[i],
                    "doc_id": ex["doc_id"],
                    "episode": ex["episode"],
                    "character_a": ex["character_a"],
                    "character_b": ex["character_b"],
                    "true_power_rating": y_rating[i],
                    "pred_power_rating": rating_pred[i],
                    "power_rating_probs": json.dumps(rating_probs[i]),
                    "true_power_shift": int(y_shift[i]),
                    "pred_power_shift": int(shift_pred[i]),
                    "power_shift_probs": json.dumps(shift_probs[i]),
                    "true_power_strategies": json.dumps(true_strategies),
                    "pred_power_strategies": json.dumps(pred_strategies),
                    "power_strategy_probs": json.dumps(strategy_probs[i]),
                    "text": ex["text"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations/all_annotations.json")
    parser.add_argument("--dialogues", default="dialogues")
    parser.add_argument("--dataset", default="modeling/dataset.json")
    parser.add_argument("--model-output", default="modeling/artifacts/power_models.joblib")
    parser.add_argument("--metrics-output", default="modeling/artifacts/metrics.json")
    parser.add_argument("--predictions-output", default="modeling/artifacts/baseline_predictions.csv")
    parser.add_argument("--model-name", default="tfidf_logreg_baseline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--skip-missing-dialogues", action="store_true")
    args = parser.parse_args()

    examples = load_or_build_examples(args)
    if len(examples) < 25:
        raise ValueError("Need more examples for a useful train/test split.")

    texts = np.array(
        [
            (
                f"Character A: {ex['character_a']}\n"
                f"Character B: {ex['character_b']}\n\n"
                f"{ex['text']}"
            )
            for ex in examples
        ],
        dtype=object,
    )
    y_rating = np.array([ex["power_rating"] for ex in examples], dtype=object)
    y_shift = np.array([ex["power_shift"] for ex in examples], dtype=int)
    strategy_sets = [ex["power_strategies"] for ex in examples]
    mlb = MultiLabelBinarizer(classes=STRATEGIES)
    y_strategies = mlb.fit_transform(strategy_sets)

    train_idx, test_idx = train_test_indices(y_rating, args.test_size, args.seed)
    x_train, x_test = texts[train_idx], texts[test_idx]
    split_names = np.array(["train"] * len(examples), dtype=object)
    split_names[test_idx] = "test"

    rating_model = make_text_classifier(class_weight="balanced")
    shift_model = make_text_classifier(class_weight="balanced")
    strategy_model = make_strategy_classifier()

    rating_model.fit(x_train, y_rating[train_idx])
    if len(set(y_shift[train_idx])) < 2:
        shift_model = DummyClassifier(strategy="most_frequent")
    shift_model.fit(x_train, y_shift[train_idx])
    strategy_model.fit(x_train, y_strategies[train_idx])

    rating_pred = rating_model.predict(x_test)
    shift_pred = shift_model.predict(x_test)
    strategy_pred = strategy_model.predict(x_test)

    metrics = {
        "n_examples": len(examples),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "rating": evaluate_single_task(
            "Power rating", y_rating[test_idx], rating_pred, RATING_LABELS
        ),
        "power_shift": evaluate_single_task(
            "Power shift", y_shift[test_idx], shift_pred, [0, 1]
        ),
        "strategies": {
            "micro_f1": f1_score(
                y_strategies[test_idx], strategy_pred, average="micro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_strategies[test_idx], strategy_pred, average="macro", zero_division=0
            ),
            "hamming_loss": hamming_loss(y_strategies[test_idx], strategy_pred),
            "classification_report": classification_report(
                y_strategies[test_idx],
                strategy_pred,
                target_names=STRATEGIES,
                zero_division=0,
            ),
        },
    }
    print("\nPower strategies")
    print(metrics["strategies"]["classification_report"])
    print(
        "micro_f1={micro_f1:.3f} macro_f1={macro_f1:.3f} hamming_loss={hamming_loss:.3f}".format(
            **metrics["strategies"]
        )
    )

    artifact = {
        "rating_model": rating_model,
        "shift_model": shift_model,
        "strategy_model": strategy_model,
        "strategy_binarizer": mlb,
        "rating_labels": RATING_LABELS,
        "strategy_labels": STRATEGIES,
        "seed": args.seed,
    }
    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)

    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_output = Path(args.predictions_output)
    save_predictions(
        output_path=predictions_output,
        model_name=args.model_name,
        examples=examples,
        split_names=split_names,
        texts=texts,
        rating_model=rating_model,
        shift_model=shift_model,
        strategy_model=strategy_model,
        y_rating=y_rating,
        y_shift=y_shift,
        y_strategies=y_strategies,
    )
    print(f"\nSaved model to {model_output}")
    print(f"Saved metrics to {metrics_output}")
    print(f"Saved predictions to {predictions_output}")


if __name__ == "__main__":
    main()
