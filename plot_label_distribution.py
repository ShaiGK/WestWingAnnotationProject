#!/usr/bin/env python3
"""
plot_label_distribution.py — Stacked bar chart of power_rating distribution
per annotator across all their annotations (not just IAA items).

Output: figures/label_distribution.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANNOTATORS = ["claire", "evan", "galileo", "nathan", "quinn", "shai"]
RATING_LABELS = ["-2", "-1", "0", "+1", "+2"]
# Diverging red→blue palette matching compute_iaa.py
COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]

SPLIT_DIR = Path("annotations/split_annotations")
FIGURES_DIR = Path("figures")


def load_counts():
    counts = {}
    for name in ANNOTATORS:
        path = SPLIT_DIR / f"{name}_annotations.json"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping {name}")
            continue
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        c = Counter(r.get("power_rating") for r in records)
        counts[name] = {label: c.get(label, 0) for label in RATING_LABELS}
    return counts


def plot(counts):
    annotators = list(counts.keys())
    n = len(annotators)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))

    totals = np.array([sum(counts[a][l] for l in RATING_LABELS) for a in annotators])

    bottoms = np.zeros(n)
    bars = []
    for label, color in zip(RATING_LABELS, COLORS):
        vals = np.array([counts[a][label] / totals[i] * 100 if totals[i] else 0
                         for i, a in enumerate(annotators)])
        bar = ax.bar(x, vals, bottom=bottoms, color=color,
                     edgecolor="white", linewidth=0.5, label=label)
        bars.append((bar, vals, bottoms.copy()))
        bottoms += vals

    # Percentage labels inside segments (skip if segment is too small)
    for bar, vals, bot in bars:
        for rect, v, b in zip(bar, vals, bot):
            if v < 5:
                continue
            cx = rect.get_x() + rect.get_width() / 2
            cy = b + v / 2
            ax.text(cx, cy, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a.capitalize()}\n(n={int(totals[i])})"
                        for i, a in enumerate(annotators)])
    ax.set_xlabel("Annotator", fontsize=11)
    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Power Rating Distribution per Annotator  (All Annotations)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Rating", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "label_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    counts = load_counts()
    if not counts:
        print("No annotation files found. Run processing.py first to generate split files.")
    else:
        plot(counts)
