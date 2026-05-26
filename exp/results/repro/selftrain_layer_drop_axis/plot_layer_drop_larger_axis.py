#!/usr/bin/env python3
"""Regenerate the layer-drop self-training figure with a wider WER axis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "layer_drop_lr_sweep_9e-5_source.csv"
DEFAULT_OUTPUT = ROOT / "layer_drop_lr_sweep_ablation_bars_larger_axis.pdf"
DEFAULT_PNG = ROOT / "layer_drop_lr_sweep_ablation_bars_larger_axis.png"

SETTING_ORDER = [
    "drop-none",
    "drop-subsampling",
    "drop-layer-0",
    "drop-layer-1",
    "drop-layer-2",
    "drop-layer-3",
    "drop-layer-4",
    "drop-layer-5",
    "drop-ctc-decoder",
]

SETTING_LABELS = {
    "drop-none": "none",
    "drop-subsampling": "sub.",
    "drop-layer-0": "L0",
    "drop-layer-1": "L1",
    "drop-layer-2": "L2",
    "drop-layer-3": "L3",
    "drop-layer-4": "L4",
    "drop-layer-5": "L5",
    "drop-ctc-decoder": "ctc dec.",
}

DATASET_TITLES = {
    "earnings22": "Earnings22",
    "tedlium": "TED-LIUM",
}

DATASET_ORDER = ["earnings22", "tedlium"]

BASELINE_WER = {
    "earnings22": 0.18289320507321855,
    "tedlium": 0.06227184121920964,
}


def copy_source_rows(source_summary: Path, output_csv: Path) -> None:
    """Copy only the layer-drop lr=9e-5 rows used by this figure."""
    with source_summary.open(newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row.get("group") == "layer_drop_lr_sweep"
            and row.get("lr") == "9em5"
            and not row.get("error")
        ]
    if not rows:
        raise SystemExit(f"No layer_drop_lr_sweep lr=9em5 rows found in {source_summary}")

    rows.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99,
            SETTING_ORDER.index(row["setting"]) if row["setting"] in SETTING_ORDER else 99,
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(source_csv: Path) -> list[dict[str, str]]:
    with source_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("wer") and not row.get("error")]


def mean_relative_axis_limits(values: list[float]) -> tuple[float, float]:
    mean_value = float(np.mean(values))
    return mean_value * 0.8, mean_value * 1.2


def render(source_csv: Path, output_pdf: Path, output_png: Path | None) -> None:
    rows = load_rows(source_csv)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.8),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, dataset in zip(axes[0], DATASET_ORDER):
        subset = [row for row in rows if row["dataset"] == dataset]
        if not subset:
            raise SystemExit(f"No rows found for dataset={dataset}")

        by_setting = {row["setting"]: float(row["wer"]) * 100.0 for row in subset}
        settings = [setting for setting in SETTING_ORDER if setting in by_setting]
        values = [by_setting[setting] for setting in settings]
        x = np.arange(len(settings))

        bars = ax.bar(
            x,
            values,
            width=0.68,
            color="#DD8452",
            edgecolor="black",
            linewidth=0.55,
            label="lr=9e-5",
        )

        axis_bottom, axis_top = mean_relative_axis_limits(values)
        label_offset = (axis_top - axis_bottom) * 0.02
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        baseline = BASELINE_WER[dataset] * 100.0
        ax.text(
            0.98,
            0.96,
            f"unadapted WER={baseline:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#C44E52",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

        ax.set_ylim(axis_bottom, axis_top)
        ax.set_title(DATASET_TITLES[dataset], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SETTING_LABELS.get(setting, setting) for setting in settings],
            fontsize=8,
            rotation=35,
            ha="right",
        )
        ax.set_ylabel("WER (%)")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()

    if args.source_summary is not None:
        copy_source_rows(args.source_summary, args.source_csv)
    render(args.source_csv, args.output, args.png)
    print(f"Saved {args.output}")
    print(f"Saved {args.png}")
    print(f"Source rows: {args.source_csv}")


if __name__ == "__main__":
    main()
