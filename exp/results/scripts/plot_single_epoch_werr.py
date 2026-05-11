#!/usr/bin/env python3
"""Plot relative WER reduction for learnt augmentation results.

The script parses checked-in result text files and does not use paper-table
fallback values. Missing result files or missing epoch entries are left blank in
the CSV and omitted from the bar chart.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


RESULT_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s*-\s*"
    r"Split:\s*(?P<split>\S+)\s*-\s*"
    r"Epochs:\s*(?P<epochs>\d+)\s*-\s*"
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*"
    r"Updated_WER:\s*(?P<updated>[0-9.]+)"
)


@dataclass(frozen=True)
class ResultSpec:
    dataset_key: str
    label: str
    method: str
    path: Path
    epoch: int


DATASETS = [
    ("tedlium", "TED-LIUM"),
    ("earnings22", "Earnings-22"),
    ("rev16", "Rev16"),
    ("this_american_life", "TAL"),
    ("chime6", "CHiME-6"),
]

METHODS = ["UFMR", "Random FM"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "historical_results"


def result_specs(results_dir: Path, epoch: int) -> list[ResultSpec]:
    ufmr_dir = "single_epoch" if epoch == 1 else "multiepoch"
    rfm_dir = "singleepoch" if epoch == 1 else "multiepoch"

    return [
        ResultSpec("tedlium", "TED-LIUM", "UFMR", results_dir / f"UFMR/{ufmr_dir}/tedlium.txt", epoch),
        ResultSpec("earnings22", "Earnings-22", "UFMR", results_dir / f"UFMR/{ufmr_dir}/e22.txt", epoch),
        ResultSpec("rev16", "Rev16", "UFMR", results_dir / f"UFMR/{ufmr_dir}/rev16.txt", epoch),
        ResultSpec("this_american_life", "TAL", "UFMR", results_dir / f"UFMR/{ufmr_dir}/TAL.txt", epoch),
        ResultSpec("chime6", "CHiME-6", "UFMR", results_dir / f"UFMR/{ufmr_dir}/chime6.txt", epoch),
        ResultSpec("tedlium", "TED-LIUM", "Random FM", results_dir / f"RFM/{rfm_dir}/tedlium.txt", epoch),
        ResultSpec("earnings22", "Earnings-22", "Random FM", results_dir / f"RFM/{rfm_dir}/e22.txt", epoch),
        ResultSpec("rev16", "Rev16", "Random FM", results_dir / f"RFM/{rfm_dir}/rev16.txt", epoch),
        ResultSpec("this_american_life", "TAL", "Random FM", results_dir / f"RFM/{rfm_dir}/TAL.txt", epoch),
        ResultSpec("chime6", "CHiME-6", "Random FM", results_dir / f"RFM/{rfm_dir}/chime6.txt", epoch),
    ]


def parse_result(path: Path, dataset_key: str, epoch: int) -> tuple[float, float] | None:
    if not path.exists():
        return None

    fallback: tuple[float, float] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        if match.group("dataset") != dataset_key:
            continue
        if int(match.group("epochs")) != epoch:
            continue

        parsed = float(match.group("original")), float(match.group("updated"))
        if match.group("split") == "test":
            return parsed
        if fallback is None:
            fallback = parsed

    return fallback


def relative_reduction(original: float, updated: float) -> float:
    return 100.0 * (original - updated) / original


def collect_results(results_dir: Path, epoch: int) -> dict[str, dict[str, dict[str, object]]]:
    rows: dict[str, dict[str, dict[str, object]]] = {
        dataset_key: {} for dataset_key, _ in DATASETS
    }

    for spec in result_specs(results_dir, epoch):
        parsed = parse_result(spec.path, spec.dataset_key, spec.epoch)
        if parsed is None:
            rows[spec.dataset_key][spec.method] = {
                "label": spec.label,
                "epoch": spec.epoch,
                "source": "",
                "original": None,
                "updated": None,
                "werr": None,
            }
            continue

        original, updated = parsed
        rows[spec.dataset_key][spec.method] = {
            "label": spec.label,
            "epoch": spec.epoch,
            "source": str(spec.path),
            "original": original,
            "updated": updated,
            "werr": relative_reduction(original, updated),
        }

    return rows


def write_csv(rows: dict[str, dict[str, dict[str, object]]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "method", "epoch", "original_wer", "updated_wer", "relative_werr", "source"])
        for dataset_key, label in DATASETS:
            for method in METHODS:
                entry = rows[dataset_key].get(method, {})
                writer.writerow(
                    [
                        label,
                        method,
                        entry.get("epoch", ""),
                        "" if entry.get("original") is None else f"{entry['original']:.10f}",
                        "" if entry.get("updated") is None else f"{entry['updated']:.10f}",
                        "" if entry.get("werr") is None else f"{entry['werr']:.4f}",
                        entry.get("source", ""),
                    ]
                )


def plot(rows: dict[str, dict[str, dict[str, object]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(DATASETS))
    width = 0.34
    offsets = {"UFMR": -width / 2, "Random FM": width / 2}
    colors = {"UFMR": "#3b6ea8", "Random FM": "#d6862b"}

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    for method in METHODS:
        label_added = False
        for idx, (dataset_key, _) in enumerate(DATASETS):
            entry = rows[dataset_key].get(method, {})
            value = entry.get("werr")
            updated_wer = entry.get("updated")
            if value is None or updated_wer is None:
                continue
            xpos = x[idx] + offsets[method]
            ax.bar(
                xpos,
                value,
                width,
                label=method if not label_added else None,
                color=colors[method],
                edgecolor="black",
                linewidth=0.4,
            )
            label_added = True
            ax.text(xpos, value + 0.5, f"{updated_wer * 100:.1f}", ha="center", va="bottom", fontsize=8)

    max_value = max(
        (entry["werr"] for dataset_rows in rows.values() for entry in dataset_rows.values() if entry.get("werr") is not None),
        default=1.0,
    )
    ax.set_ylabel("Relative WER reduction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS], rotation=20, ha="right")
    ax.set_ylim(0, max(5.0, max_value + 4.0))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    output_stem = "single_epoch_werr" if args.epoch == 1 else f"epoch{args.epoch}_werr"
    output_path = args.output or args.results_dir / "figures" / f"{output_stem}.pdf"
    csv_path = args.csv or args.results_dir / "figures" / f"{output_stem}.csv"

    rows = collect_results(args.results_dir, args.epoch)
    write_csv(rows, csv_path)
    plot(rows, output_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
