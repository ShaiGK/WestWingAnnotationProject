#!/usr/bin/env python3
"""Build model-ready examples from annotation rows and dialogue files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RATING_LABELS = ["-2", "-1", "0", "+1", "+2"]
RATING_TO_INT = {"-2": -2, "-1": -1, "0": 0, "+1": 1, "+2": 2}
STRATEGIES = [
    "Direct orders or instructions",
    "Controls information",
    "Dismisses or shuts down",
    "Interrogates or corners",
    "Appeals to authority or rank",
    "Humor or sarcasm to assert",
    "Manages or caretakes",
    "Emotional pressure or reprimand",
]


def normalize_rating(value: Any) -> str:
    if value is None:
        raise ValueError("Missing power_rating")
    text = str(value).strip()
    if text in RATING_LABELS:
        return text
    if text in {"1", "2"}:
        return f"+{text}"
    raise ValueError(f"Unknown power_rating value: {value!r}")


def doc_id_to_dialogue_path(doc_id: str, dialogues_dir: Path) -> Path:
    """Map S01E01_A-B_pair1_exc2 to dialogues/season_1/S01E01_A-B_01_02.txt."""
    match = re.match(r"^(S(\d+)E\d+)_(.+)_pair(\d+)_exc(\d+)$", doc_id)
    if not match:
        raise ValueError(f"Cannot parse doc_id: {doc_id}")
    episode, season, pair_name, pair_instance, excerpt = match.groups()
    filename = f"{episode}_{pair_name}_{int(pair_instance):02d}_{int(excerpt):02d}.txt"
    return dialogues_dir / f"season_{int(season)}" / filename


def _filename_safe_pair_name(pair_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", pair_name)


def resolve_dialogue_path(doc_id: str, dialogues_dir: Path) -> Path:
    path = doc_id_to_dialogue_path(doc_id, dialogues_dir)
    if path.exists():
        return path

    match = re.match(r"^(S(\d+)E\d+)_(.+)_pair(\d+)_exc(\d+)$", doc_id)
    if not match:
        return path
    episode, season, pair_name, pair_instance, excerpt = match.groups()
    safe_name = _filename_safe_pair_name(pair_name)
    safe_path = (
        dialogues_dir
        / f"season_{int(season)}"
        / f"{episode}_{safe_name}_{int(pair_instance):02d}_{int(excerpt):02d}.txt"
    )
    if safe_path.exists():
        return safe_path

    candidates = list(
        (dialogues_dir / f"season_{int(season)}").glob(
            f"{episode}_*_{int(pair_instance):02d}_{int(excerpt):02d}.txt"
        )
    )
    normalized_pair = _filename_safe_pair_name(pair_name).upper()
    matches = [
        candidate
        for candidate in candidates
        if candidate.name.split("_", 2)[1].upper() == normalized_pair
    ]
    if len(matches) == 1:
        return matches[0]
    return path


def read_dialogue_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "---DIALOGUE---"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


def _majority_vote(values: list[Any]) -> Any:
    counts = Counter(values)
    top_count = counts.most_common(1)[0][1]
    winners = [value for value, count in counts.items() if count == top_count]
    if len(winners) == 1:
        return winners[0]
    # Tie-break ordinal ratings toward the mean annotator judgment.
    if all(value in RATING_TO_INT for value in winners):
        mean_value = sum(RATING_TO_INT[v] for v in values) / len(values)
        return min(winners, key=lambda v: (abs(RATING_TO_INT[v] - mean_value), abs(RATING_TO_INT[v])))
    return sorted(winners, key=str)[0]


def aggregate_annotations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse multiple annotators for the same excerpt into one training target."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["doc_id"]].append(row)

    aggregated = {}
    for doc_id, anns in grouped.items():
        ratings = [normalize_rating(a["power_rating"]) for a in anns]
        shifts = [1 if a.get("power_shift") == "Yes" else 0 for a in anns]
        strategy_counts = Counter(
            strategy
            for ann in anns
            for strategy in (ann.get("power_strategies") or [])
        )
        threshold = len(anns) / 2

        first = anns[0]
        aggregated[doc_id] = {
            "doc_id": doc_id,
            "episode": first.get("episode"),
            "character_a": first.get("character_a"),
            "character_b": first.get("character_b"),
            "pair_instance": first.get("pair_instance"),
            "excerpt": first.get("excerpt"),
            "n_annotators": len(anns),
            "power_rating": _majority_vote(ratings),
            "power_shift": int(sum(shifts) > threshold),
            "power_strategies": [
                strategy for strategy in STRATEGIES if strategy_counts[strategy] > threshold
            ],
            "annotator_ratings": ratings,
        }
    return aggregated


def build_dataset(
    annotations_path: Path,
    dialogues_dir: Path,
    output_path: Path | None = None,
    skip_missing_dialogues: bool = False,
) -> list[dict[str, Any]]:
    rows = json.loads(annotations_path.read_text(encoding="utf-8"))
    aggregated = aggregate_annotations(rows)
    examples = []
    missing = []

    for doc_id, example in sorted(aggregated.items()):
        dialogue_path = resolve_dialogue_path(doc_id, dialogues_dir)
        if not dialogue_path.exists():
            missing.append((doc_id, str(dialogue_path)))
            if skip_missing_dialogues:
                continue
            raise FileNotFoundError(f"Missing dialogue file for {doc_id}: {dialogue_path}")

        text = read_dialogue_text(dialogue_path)
        examples.append(
            {
                **example,
                "dialogue_path": str(dialogue_path),
                "text": text,
                "strategy_labels": {
                    strategy: int(strategy in example["power_strategies"])
                    for strategy in STRATEGIES
                },
            }
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")

    if missing:
        print(f"Skipped {len(missing)} examples with missing dialogue files.")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations/all_annotations.json")
    parser.add_argument("--dialogues", default="dialogues")
    parser.add_argument("--output", default="modeling/dataset.json")
    parser.add_argument("--skip-missing-dialogues", action="store_true")
    args = parser.parse_args()

    examples = build_dataset(
        annotations_path=Path(args.annotations),
        dialogues_dir=Path(args.dialogues),
        output_path=Path(args.output),
        skip_missing_dialogues=args.skip_missing_dialogues,
    )
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
