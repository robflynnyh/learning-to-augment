#!/usr/bin/env python3
"""Plot oracle WER against augmentation search breadth."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


RESULT_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s*-\s*"
    r"Split:\s*(?P<split>\S+)\s*-\s*"
    r"Epochs:\s*(?P<epochs>\d+)\s*-\s*"
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*"
    r"Updated_WER:\s*(?P<updated>[0-9.]+)\s*-\s*"
    r"Repeats:\s*(?P<repeats>\d+)\s*-\s*"
    r"Rollout Type:\s*(?P<rollout_type>\S+)"
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "RMM/oracle/tedlium.txt"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "figures/oracle_wer.pdf"
DEFAULT_CSV = DEFAULT_RESULTS_DIR / "figures/oracle_wer.csv"


def parse_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "dataset": match.group("dataset"),
                "split": match.group("split"),
                "epochs": int(match.group("epochs")),
                "original_wer": float(match.group("original")),
                "updated_wer": float(match.group("updated")),
                "repeats": int(match.group("repeats")),
                "rollout_type": match.group("rollout_type"),
            }
        )
    return sorted(rows, key=lambda row: row["repeats"])


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "split", "epochs", "repeats", "original_wer", "updated_wer", "rollout_type"])
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["split"],
                    row["epochs"],
                    row["repeats"],
                    f"{row['original_wer']:.10f}",
                    f"{row['updated_wer']:.10f}",
                    row["rollout_type"],
                ]
            )


def plot(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No oracle results were parsed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repeats = [row["repeats"] for row in rows]
    wers = [100.0 * row["updated_wer"] for row in rows]
    original_wer = 100.0 * rows[0]["original_wer"]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(repeats, wers, label="Oracle WER", marker="o", markersize=3.0, linewidth=1.2)
    ax.axhline(original_wer, color="black", linestyle="--", linewidth=0.8, label="No adaptation")
    ax.set_xlabel("Augmentation search breadth per step")
    ax.set_ylabel("WER")
    ax.grid(True, which="both", linewidth=0.5, color="gray", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = parse_results(args.input)
    write_csv(rows, args.csv)
    plot(rows, args.output)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
