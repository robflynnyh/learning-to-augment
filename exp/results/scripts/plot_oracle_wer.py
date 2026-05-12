#!/usr/bin/env python3
"""Plot oracle WER against augmentation search breadth."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".scratch" / "matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


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
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "historical_results"
DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "RMM/oracle/tedlium.txt"
DEFAULT_UFMR_INPUT = DEFAULT_RESULTS_DIR / "UFMR_segmented/tedlium.txt"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "figures/oracle_wer.pdf"
DEFAULT_CSV = DEFAULT_RESULTS_DIR / "figures/oracle_wer.csv"


def parse_series_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("series must be formatted as label=path")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("series label cannot be empty")
    return label.strip(), Path(path)


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


def write_csv(series: list[tuple[str, list[dict[str, object]]]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["series", "dataset", "split", "epochs", "repeats", "original_wer", "updated_wer", "rollout_type"])
        for label, rows in series:
            for row in rows:
                writer.writerow(
                    [
                        label,
                        row["dataset"],
                        row["split"],
                        row["epochs"],
                        row["repeats"],
                        f"{row['original_wer']:.10f}",
                        f"{row['updated_wer']:.10f}",
                        row["rollout_type"],
                    ]
                )


def plot(
    series: list[tuple[str, list[dict[str, object]]]],
    output_path: Path,
    ufmr_rows: list[dict[str, object]] | None = None,
    ufmr_label: str = "UFMR",
    log_x: bool = False,
) -> None:
    if not series or not series[0][1]:
        raise ValueError("No oracle results were parsed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_repeats = sorted({row["repeats"] for _, rows in series for row in rows})
    original_wer = 100.0 * series[0][1][0]["original_wer"]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#56B4E9",
        "#E69F00",
        "#000000",
    ]
    for idx, (label, rows) in enumerate(series):
        if not rows:
            continue
        repeats = [row["repeats"] for row in rows]
        wers = [100.0 * row["updated_wer"] for row in rows]
        ax.plot(
            repeats,
            wers,
            label=label,
            marker="o",
            markersize=3.4,
            linewidth=1.5,
            color=colors[idx % len(colors)],
        )
    ax.axhline(original_wer, color="#222222", linestyle="--", linewidth=1.0, label="No adaptation")
    if ufmr_rows:
        ufmr_wer = 100.0 * ufmr_rows[-1]["updated_wer"]
        ax.axhline(ufmr_wer, color="#666666", linestyle=":", linewidth=1.2, label=ufmr_label)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(min(all_repeats) * 0.9, max(all_repeats) * 1.1)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
    ax.set_xticks(all_repeats)
    ax.set_xlabel("Oracle search repeats" + (" (log scale)" if log_x else ""))
    ax.set_ylabel("WER (%)")
    ax.grid(True, which="both", linewidth=0.5, color="gray", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--label", default="Oracle WER")
    parser.add_argument(
        "--series",
        action="append",
        type=parse_series_arg,
        help="Additional or replacement oracle series formatted as label=path. Can be repeated.",
    )
    parser.add_argument("--ufmr-input", type=Path, default=DEFAULT_UFMR_INPUT)
    parser.add_argument("--ufmr-label", default="UFMR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--log-x", action="store_true", help="Use a log-scaled repeat axis.")
    args = parser.parse_args()

    series_specs = args.series if args.series else [(args.label, args.input)]
    series = [(label, parse_results(path)) for label, path in series_specs]
    ufmr_rows = parse_results(args.ufmr_input) if args.ufmr_input.exists() else []
    write_csv(series + ([(args.ufmr_label, ufmr_rows)] if ufmr_rows else []), args.csv)
    plot(series, args.output, ufmr_rows=ufmr_rows, ufmr_label=args.ufmr_label, log_x=args.log_x)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")
    for label, path in series_specs:
        print(f"Added series {label} from {path}")
    if ufmr_rows:
        print(f"Added UFMR line from {args.ufmr_input}")


if __name__ == "__main__":
    main()
