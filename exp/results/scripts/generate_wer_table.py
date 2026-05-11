#!/usr/bin/env python3
"""Generate a LaTeX WER table from learnt-augmentation result logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


RESULT_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s*-\s*"
    r"Split:\s*(?P<split>\S+)\s*-\s*"
    r"Epochs:\s*(?P<epochs>\d+)\s*-\s*"
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*"
    r"Updated_WER:\s*(?P<updated>[0-9.]+)"
)


@dataclass(frozen=True)
class ResultSpec:
    method: str
    epoch_label: str
    dataset_key: str
    path: Path
    epoch: int | None
    value: str


DATASETS = [
    ("tedlium", "TED-LIUM"),
    ("earnings22", "Earnings-22"),
    ("rev16", "Rev16"),
    ("this_american_life", "TAL"),
    ("chime6", "CHiME-6"),
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "historical_results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "figures/learnt_self_training_wer_table.txt"


def parse_result(path: Path, dataset_key: str, epoch: int | None, value: str) -> float | None:
    if not path.exists():
        return None

    fallback: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        if match.group("dataset") != dataset_key:
            continue
        if epoch is not None and int(match.group("epochs")) != epoch:
            continue

        parsed = float(match.group(value))
        if match.group("split") == "test":
            return parsed
        if fallback is None:
            fallback = parsed

    return fallback


def specs(results_dir: Path) -> list[ResultSpec]:
    return [
        ResultSpec("UFMR", "1", "tedlium", results_dir / "UFMR/single_epoch/tedlium.txt", 1, "updated"),
        ResultSpec("UFMR", "1", "earnings22", results_dir / "UFMR/single_epoch/e22.txt", 1, "updated"),
        ResultSpec("UFMR", "1", "rev16", results_dir / "UFMR/single_epoch/rev16.txt", 1, "updated"),
        ResultSpec("UFMR", "1", "this_american_life", results_dir / "UFMR/single_epoch/TAL.txt", 1, "updated"),
        ResultSpec("UFMR", "1", "chime6", results_dir / "UFMR/single_epoch/chime6.txt", 1, "updated"),
        ResultSpec("UFMR", "5", "tedlium", results_dir / "UFMR/multiepoch/tedlium.txt", 5, "updated"),
        ResultSpec("UFMR", "5", "earnings22", results_dir / "UFMR/multiepoch/e22.txt", 5, "updated"),
        ResultSpec("UFMR", "5", "rev16", results_dir / "UFMR/multiepoch/rev16.txt", 5, "updated"),
        ResultSpec("UFMR", "5", "this_american_life", results_dir / "UFMR/multiepoch/TAL.txt", 5, "updated"),
        ResultSpec("UFMR", "5", "chime6", results_dir / "UFMR/multiepoch/chime6.txt", 5, "updated"),
        ResultSpec("Random FM", "1", "tedlium", results_dir / "RFM/singleepoch/tedlium.txt", 1, "updated"),
        ResultSpec("Random FM", "1", "earnings22", results_dir / "RFM/singleepoch/e22.txt", 1, "updated"),
        ResultSpec("Random FM", "1", "rev16", results_dir / "RFM/singleepoch/rev16.txt", 1, "updated"),
        ResultSpec("Random FM", "1", "this_american_life", results_dir / "RFM/singleepoch/TAL.txt", 1, "updated"),
        ResultSpec("Random FM", "1", "chime6", results_dir / "RFM/singleepoch/chime6.txt", 1, "updated"),
        ResultSpec("Random FM", "5", "tedlium", results_dir / "RFM/multiepoch/tedlium.txt", 5, "updated"),
        ResultSpec("Random FM", "5", "earnings22", results_dir / "RFM/multiepoch/e22.txt", 5, "updated"),
        ResultSpec("Random FM", "5", "rev16", results_dir / "RFM/multiepoch/rev16.txt", 5, "updated"),
        ResultSpec("Random FM", "5", "this_american_life", results_dir / "RFM/multiepoch/TAL.txt", 5, "updated"),
        ResultSpec("Random FM", "5", "chime6", results_dir / "RFM/multiepoch/chime6.txt", 5, "updated"),
        ResultSpec("No augmentation", "3", "tedlium", results_dir / "NoAug/tedlium.txt", 3, "updated"),
        ResultSpec("No augmentation", "3", "earnings22", results_dir / "NoAug/earnings22.txt", 3, "updated"),
        ResultSpec("No augmentation", "3", "rev16", results_dir / "NoAug/rev16.txt", 3, "updated"),
        ResultSpec("No augmentation", "3", "this_american_life", results_dir / "NoAug/TAL.txt", 3, "updated"),
        ResultSpec("No augmentation", "3", "chime6", results_dir / "NoAug/chime6.txt", 3, "updated"),
        ResultSpec("No adaptation", "N/A", "tedlium", results_dir / "UFMR/single_epoch/tedlium.txt", 1, "original"),
        ResultSpec("No adaptation", "N/A", "earnings22", results_dir / "UFMR/single_epoch/e22.txt", 1, "original"),
        ResultSpec("No adaptation", "N/A", "rev16", results_dir / "UFMR/single_epoch/rev16.txt", 1, "original"),
        ResultSpec("No adaptation", "N/A", "this_american_life", results_dir / "UFMR/single_epoch/TAL.txt", 1, "original"),
        ResultSpec("No adaptation", "N/A", "chime6", results_dir / "UFMR/single_epoch/chime6.txt", 1, "original"),
    ]


def collect_values(results_dir: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    rows: dict[tuple[str, str], dict[str, float | None]] = {}
    for spec in specs(results_dir):
        key = (spec.method, spec.epoch_label)
        rows.setdefault(key, {})
        rows[key][spec.dataset_key] = parse_result(spec.path, spec.dataset_key, spec.epoch, spec.value)
    return rows


def format_wer(value: float | None, bold: bool) -> str:
    if value is None:
        return ""
    text = f"{value * 100:.1f}"
    return rf"\textbf{{{text}}}" if bold else text


def render_table(rows: dict[tuple[str, str], dict[str, float | None]]) -> str:
    row_order = [
        ("UFMR", "1"),
        ("UFMR", "5"),
        ("Random FM", "1"),
        ("Random FM", "5"),
        ("No augmentation", "3"),
        ("No adaptation", "N/A"),
    ]
    bold_rows = [key for key in row_order if key[0] != "No adaptation"]
    best: dict[str, float] = {}
    for dataset_key, _ in DATASETS:
        candidates = [
            rows.get(key, {}).get(dataset_key)
            for key in bold_rows
            if rows.get(key, {}).get(dataset_key) is not None
        ]
        if candidates:
            best[dataset_key] = min(candidates)

    lines = [
        r"\begin{table}[hbt]",
        r"    \centering",
        r"    \begin{tabular}{llccccc}",
        r"        \toprule",
        r"        Method & Epochs & TED-LIUM & Earnings-22 & Rev16 & TAL & CHiME-6 \\",
        r"        \midrule",
    ]

    for row_idx, key in enumerate(row_order):
        if row_idx in {2, 4, 5}:
            lines.append(r"        \midrule")
        method, epoch_label = key
        values = []
        for dataset_key, _ in DATASETS:
            value = rows.get(key, {}).get(dataset_key)
            should_bold = value is not None and dataset_key in best and abs(value - best[dataset_key]) < 1e-12
            values.append(format_wer(value, should_bold))
        lines.append(f"        {method} & {epoch_label} & " + " & ".join(values) + r" \\")

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{WERs for learnt and random frequency masking across single- and multi-epoch test-time self-training.}",
            r"    \label{tab:selftrain:learnt-augmentation-wer}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = collect_values(args.results_dir)
    table = render_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
