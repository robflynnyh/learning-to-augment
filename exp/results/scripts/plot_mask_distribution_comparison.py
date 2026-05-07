#!/usr/bin/env python3
"""Overlay two or more saved frequency-mask distributions."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent


def load_distribution(path: Path) -> tuple[list[int], list[float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["frequency_bin"]) for row in rows], [float(row["mask_probability"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Series label and CSV path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "figures/mask_distribution_comparison.pdf",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    styles = [
        {"color": "#3b6ea8", "linewidth": 1.8, "linestyle": "-"},
        {"color": "#4c4c4c", "linewidth": 1.6, "linestyle": "--"},
        {"color": "#d6862b", "linewidth": 1.6, "linestyle": "-."},
    ]

    max_probability = 0.0
    for idx, (label, csv_path) in enumerate(args.series):
        bins, probabilities = load_distribution(Path(csv_path))
        max_probability = max(max_probability, max(probabilities))
        ax.plot(bins, probabilities, label=label, **styles[idx % len(styles)])

    ax.set_xlabel("Mel-frequency bin")
    ax.set_ylabel("Mask probability")
    ax.set_xlim(-0.5, max(bins) + 0.5)
    ax.set_ylim(0.0, max(0.05, max_probability * 1.12))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
