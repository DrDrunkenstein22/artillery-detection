"""
src/evaluation/compare_models.py

Generates comparison plots and a markdown report from results/metrics/comparison.json.
Run locally after modal_evaluate.py has completed.

Usage:
    python src/evaluation/compare_models.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CLASSES = [
    "artillery", "tank", "apc", "military_truck",
    "rocket_artillery", "ifv", "military_aircraft", "other_military",
]

MODEL_COLORS = {
    "YOLOv11m":       "#2196F3",   # blue
    "RT-DETR-L":      "#4CAF50",   # green
    "Faster-RCNN-R50": "#FF5722",  # orange-red
}


def load_comparison() -> dict:
    path = METRICS_DIR / "comparison.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run modal_evaluate.py first.")
        return {}
    return json.loads(path.read_text())


def plot_accuracy_comparison(data: dict):
    """Bar chart: mAP50 and mAP50-95 for each model."""
    models = [m for m in data if data[m].get("map50") is not None]
    if not models:
        print("No mAP data available yet.")
        return

    x = np.arange(len(models))
    width = 0.35

    map50 = [data[m]["map50"] for m in models]
    map50_95 = [data[m]["map50_95"] for m in models]
    colors = [MODEL_COLORS.get(m, "#888") for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@50", color=colors, alpha=0.9)
    bars2 = ax.bar(x + width / 2, map50_95, width, label="mAP@50-95",
                   color=colors, alpha=0.5, hatch="//")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("mAP", fontsize=13)
    ax.set_title("Detection Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    print("✓ accuracy_comparison.png")


def plot_speed_vs_accuracy(data: dict):
    """Scatter: FPS vs mAP@50 with model size as bubble size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for model, metrics in data.items():
        fps = metrics.get("fps")
        map50 = metrics.get("map50")
        size = metrics.get("model_size_mb", 50)
        params = metrics.get("params_M", 10)

        if fps is None:
            continue

        color = MODEL_COLORS.get(model, "#888")
        y = map50 if map50 is not None else 0.0
        ax.scatter(fps, y, s=params * 5, color=color, alpha=0.85,
                   edgecolors="black", linewidth=1.5, zorder=3)
        ax.annotate(f"{model}\n({params:.0f}M params)",
                    (fps, y), textcoords="offset points",
                    xytext=(10, 5), fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_xlabel("Inference Speed (FPS ↑)", fontsize=13)
    ax.set_ylabel("mAP@50 (↑)", fontsize=13)
    ax.set_title("Speed vs. Accuracy Trade-off\n(bubble size = parameter count)",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.axvline(x=30, color="red", linestyle="--", alpha=0.5, label="30 FPS real-time threshold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "speed_vs_accuracy.png", dpi=150)
    plt.close()
    print("✓ speed_vs_accuracy.png")


def plot_per_class_ap(data: dict):
    """Grouped bar chart: per-class AP50 for models that have it."""
    models_with_class = {m: d for m, d in data.items() if d.get("per_class_ap50")}
    if not models_with_class:
        print("No per-class AP data available.")
        return

    classes = CANONICAL_CLASSES
    n_models = len(models_with_class)
    x = np.arange(len(classes))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (model, metrics) in enumerate(models_with_class.items()):
        ap_vals = [metrics["per_class_ap50"].get(c, 0.0) for c in classes]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, ap_vals, width,
                      label=model, color=MODEL_COLORS.get(model, "#888"), alpha=0.85)

    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("AP@50", fontsize=13)
    ax.set_title("Per-Class AP@50 Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Highlight artillery
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="red", label="Artillery (priority class)")
    ax.text(0, 1.01, "⚠ Priority", ha="center", fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_ap.png", dpi=150)
    plt.close()
    print("✓ per_class_ap.png")


def plot_efficiency_radar(data: dict):
    """Radar/spider chart: multi-metric comparison."""
    models = list(data.keys())
    if len(models) < 2:
        return

    metrics_labels = ["mAP@50", "mAP@50-95", "Precision", "Recall", "Speed (norm)"]
    n = len(metrics_labels)

    # Normalize speed to [0, 1] range based on max FPS
    max_fps = max((d.get("fps") or 0) for d in data.values()) or 1

    def get_radar_vals(metrics):
        return [
            metrics.get("map50") or 0,
            metrics.get("map50_95") or 0,
            metrics.get("precision") or 0,
            metrics.get("recall") or 0,
            (metrics.get("fps") or 0) / max_fps,
        ]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        vals = get_radar_vals(data[model])
        vals += vals[:1]
        color = MODEL_COLORS.get(model, "#888")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=model)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, size=12)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison Radar\n(all metrics normalized 0–1)",
                 pad=20, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "radar_comparison.png", dpi=150)
    plt.close()
    print("✓ radar_comparison.png")


def generate_markdown_report(data: dict):
    """Write a clean markdown summary report."""
    lines = ["# Artillery Detection — Model Comparison Report\n"]
    lines.append("*Auto-generated by `src/evaluation/compare_models.py`*\n")
    lines.append("---\n")

    lines.append("## Summary Table\n")
    lines.append("| Model | Architecture | mAP@50 | mAP@50-95 | Precision | Recall | FPS | Latency (ms) | Params (M) | Size (MB) | Real-time? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for model, m in data.items():
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))

        lines.append(
            f"| **{model}** | {m.get('architecture', '—')} "
            f"| {fmt(m.get('map50'))} "
            f"| {fmt(m.get('map50_95'))} "
            f"| {fmt(m.get('precision'))} "
            f"| {fmt(m.get('recall'))} "
            f"| {fmt(m.get('fps'))} "
            f"| {fmt(m.get('latency_ms'))} "
            f"| {fmt(m.get('params_M'))} "
            f"| {fmt(m.get('model_size_mb'))} "
            f"| {'✅' if m.get('realtime') else '❌'} |"
        )

    lines.append("\n---\n")
    lines.append("## Per-Class AP@50\n")
    lines.append("Focusing on artillery (class 0) — the primary detection target.\n")

    models_with_class = {m: d for m, d in data.items() if d.get("per_class_ap50")}
    if models_with_class:
        header = "| Class | " + " | ".join(models_with_class.keys()) + " |"
        sep = "|---|" + "---|" * len(models_with_class)
        lines.append(header)
        lines.append(sep)
        for cls in CANONICAL_CLASSES:
            row = f"| {'**' + cls + '**' if cls == 'artillery' else cls} |"
            for metrics in models_with_class.values():
                v = metrics["per_class_ap50"].get(cls)
                row += f" {f'{v:.3f}' if v is not None else '—'} |"
            lines.append(row)
    else:
        lines.append("*Run evaluation to populate per-class metrics.*\n")

    lines.append("\n---\n")
    lines.append("## Plots\n")
    lines.append("- `results/plots/accuracy_comparison.png` — mAP@50 / mAP@50-95 bar chart")
    lines.append("- `results/plots/speed_vs_accuracy.png` — FPS vs accuracy scatter")
    lines.append("- `results/plots/per_class_ap.png` — per-class AP grouped bars")
    lines.append("- `results/plots/radar_comparison.png` — multi-metric radar chart")
    lines.append("- `results/plots/class_distribution.png` — dataset class balance\n")

    lines.append("---\n")
    lines.append("## Recommendation\n")

    # Auto-generate recommendation based on data
    has_data = {m for m, d in data.items() if d.get("map50") is not None}
    if has_data:
        fastest = max(has_data, key=lambda m: data[m].get("fps") or 0)
        most_accurate = max(has_data, key=lambda m: data[m].get("map50") or 0)

        if fastest == most_accurate:
            lines.append(f"**{fastest}** is both the fastest and most accurate model — clear winner for real-time artillery detection.")
        else:
            lines.append(
                f"- **Best for real-time deployment**: `{fastest}` "
                f"({data[fastest].get('fps', '?'):.0f} FPS, mAP@50={data[fastest].get('map50', 0):.3f})"
            )
            lines.append(
                f"- **Highest accuracy**: `{most_accurate}` "
                f"(mAP@50={data[most_accurate].get('map50', 0):.3f}, "
                f"{data[most_accurate].get('fps', '?'):.0f} FPS)"
            )
    else:
        lines.append("*Run evaluation to generate recommendation.*")

    report_path = RESULTS_DIR / "comparison_report.md"
    report_path.write_text("\n".join(lines))
    print(f"✓ Report: {report_path}")


def main():
    print("Loading comparison data…")
    data = load_comparison()
    if not data:
        return

    print(f"Models found: {list(data.keys())}\n")

    plot_accuracy_comparison(data)
    plot_speed_vs_accuracy(data)
    plot_per_class_ap(data)
    plot_efficiency_radar(data)
    generate_markdown_report(data)

    print(f"\n✓ All plots saved to {PLOTS_DIR}/")
    print(f"✓ Report: {RESULTS_DIR}/comparison_report.md")


if __name__ == "__main__":
    main()
