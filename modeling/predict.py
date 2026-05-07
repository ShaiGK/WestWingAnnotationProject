#!/usr/bin/env python3
"""Predict power labels for one dialogue file or raw text file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

try:
    from modeling.data import read_dialogue_text
except ModuleNotFoundError:
    from data import read_dialogue_text


def class_probs(model, text: str, labels: list) -> dict[str, float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_) if hasattr(model, "classes_") else labels
    return {str(label): float(prob) for label, prob in zip(classes, probs)}


def strategy_probs(model, text: str, labels: list) -> dict[str, float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba([text])[0]
    return {label: float(prob) for label, prob in zip(labels, probs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Dialogue .txt file to classify")
    parser.add_argument("--model", default="modeling/artifacts/power_models.joblib")
    parser.add_argument("--character-a", default="")
    parser.add_argument("--character-b", default="")
    parser.add_argument("--strategy-threshold", type=float, default=0.5)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    input_path = Path(args.input_path)
    text = read_dialogue_text(input_path)
    model_text = f"Character A: {args.character_a}\nCharacter B: {args.character_b}\n\n{text}"

    rating = artifact["rating_model"].predict([model_text])[0]
    shift = int(artifact["shift_model"].predict([model_text])[0])
    strategy_binary = artifact["strategy_model"].predict([model_text])[0]
    strategy_labels = artifact["strategy_labels"]
    strategies = [
        label for label, value in zip(strategy_labels, strategy_binary) if int(value) == 1
    ]

    probabilities = {
        "power_rating": class_probs(artifact["rating_model"], model_text, artifact["rating_labels"]),
        "power_shift": class_probs(artifact["shift_model"], model_text, [0, 1]),
        "power_strategies": strategy_probs(
            artifact["strategy_model"], model_text, strategy_labels
        ),
    }

    if probabilities["power_strategies"]:
        strategies = [
            label
            for label, prob in probabilities["power_strategies"].items()
            if prob >= args.strategy_threshold
        ]

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "power_rating": rating,
                "power_shift": bool(shift),
                "power_strategies": strategies,
                "probabilities": probabilities,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
