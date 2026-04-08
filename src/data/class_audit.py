"""
src/data/class_audit.py

Visualizes class distribution across splits.
Run this BEFORE training to verify balance.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CANONICAL_CLASSES = [
    "artillery", "tank", "apc", "military_truck",
    "rocket_artillery", "ifv", "military_aircraft", "other_military",
]


def count_labels(split_dir: Path) -> Counter:
    counts = Counter()
    lbl_dir = split_dir / "labels"
    if not lbl_dir.exists():
        return counts
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1
    return counts


def audit(splits_root: Path = Path("data/splits"), save_dir: Path = Path("results/plots")):
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    all_counts = {s: count_labels(splits_root / s) for s in splits}

    # Print table
    print(f"\n{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    for idx, name in enumerate(CANONICAL_CLASSES):
        tr = all_counts["train"][idx]
        va = all_counts["val"][idx]
        te = all_counts["test"][idx]
        print(f"{name:<25} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CANONICAL_CLASSES)))

    for ax, split in zip(axes, splits):
        counts = [all_counts[split][i] for i in range(len(CANONICAL_CLASSES))]
        bars = ax.bar(CANONICAL_CLASSES, counts, color=colors)
        ax.set_title(f"{split.capitalize()} Split", fontsize=14, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Annotation Count")
        ax.tick_params(axis="x", rotation=45)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=9)

    plt.suptitle("Artillery Detection Dataset — Class Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out}")

    # Artillery fraction warning
    total_train = sum(all_counts["train"].values())
    artillery_train = all_counts["train"][0]
    frac = artillery_train / total_train if total_train else 0
    if frac < 0.08:
        print(f"\n⚠ Artillery is only {frac:.1%} of training annotations!")
        print("  Consider increasing target_per_class for artillery in merge_datasets.py")
    else:
        print(f"\n✓ Artillery fraction in train: {frac:.1%} — looks reasonable")


if __name__ == "__main__":
    audit()
