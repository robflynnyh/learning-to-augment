#!/usr/bin/env python3
"""Summarize ROB-108 test-set policy evaluation results."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


RESULT_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s*-\s*"
    r"Split:\s*(?P<split>\S+)\s*-\s*"
    r"Epochs:\s*(?P<epochs>\d+)\s*-\s*"
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*"
    r"Updated_WER:\s*(?P<updated>[0-9.]+)"
)

DATASETS = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}
POLICY_METHODS = ("RFM", "RMM", "UFMR", "UVQLM")


def parse_result(
    path: Path,
    expected_dataset: str,
    expected_split: str,
    expected_epochs: int,
) -> tuple[float, float] | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        match = RESULT_RE.search(line)
        if not match:
            continue
        if match.group("dataset") != expected_dataset:
            continue
        if match.group("split") != expected_split:
            continue
        if int(match.group("epochs")) != expected_epochs:
            continue
        return float(match.group("original")), float(match.group("updated"))
    return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def expected_policy_cells(
    result_root: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    repeats: tuple[int, ...],
    epoch1_lrs: tuple[str, ...],
    epoch5_lrs: tuple[str, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_tag in datasets:
        dataset, split = DATASETS[dataset_tag]
        for method in methods:
            if method == "NoAug":
                for repeat in repeats:
                    repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
                    tag = f"{dataset_tag}_baseline{repeat_suffix}"
                    rows.append(
                        {
                            "dataset_tag": dataset_tag,
                            "dataset": dataset,
                            "split": split,
                            "method": method,
                            "repeat": str(repeat),
                            "seed": str(123456 + repeat - 1),
                            "epochs": "0",
                            "lr": "baseline",
                            "result_path": display_path(result_root / method / f"{tag}.txt"),
                        }
                    )
                continue
            if method not in POLICY_METHODS:
                raise ValueError(f"Unknown method: {method}")
            epoch_lrs = ((1, epoch1_lrs), (5, epoch5_lrs))
            for repeat in repeats:
                repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
                seed = 123456 + repeat - 1
                for epoch_count, lrs in epoch_lrs:
                    for lr in lrs:
                        tag = f"{dataset_tag}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
                        rows.append(
                            {
                                "dataset_tag": dataset_tag,
                                "dataset": dataset,
                                "split": split,
                                "method": method,
                                "repeat": str(repeat),
                                "seed": str(seed),
                                "epochs": str(epoch_count),
                                "lr": lr,
                                "result_path": display_path(result_root / method / f"{tag}.txt"),
                            }
                        )
    return rows


def collect_rows(
    result_root: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    repeats: tuple[int, ...],
    epoch1_lrs: tuple[str, ...],
    epoch5_lrs: tuple[str, ...],
) -> list[dict[str, str]]:
    rows = expected_policy_cells(result_root, datasets, methods, repeats, epoch1_lrs, epoch5_lrs)
    for row in rows:
        parsed = parse_result(
            Path(row["result_path"]),
            expected_dataset=row["dataset"],
            expected_split=row["split"],
            expected_epochs=int(row["epochs"]),
        )
        if parsed is None:
            row.update(
                {
                    "status": "missing",
                    "original_wer": "",
                    "updated_wer": "",
                    "absolute_delta": "",
                    "relative_delta_pct": "",
                }
            )
            continue
        original, updated = parsed
        row.update(
            {
                "status": "complete",
                "original_wer": f"{original:.6f}",
                "updated_wer": f"{updated:.6f}",
                "absolute_delta": f"{updated - original:.6f}",
                "relative_delta_pct": f"{((updated - original) / original) * 100:.2f}" if original else "",
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        key = (row["dataset_tag"], row["method"], row["epochs"], row["lr"])
        grouped.setdefault(key, []).append(row)

    aggregate: list[dict[str, str]] = []
    for (dataset_tag, method, epochs, lr), group in sorted(grouped.items()):
        original_values = [float(row["original_wer"]) for row in group]
        updated_values = [float(row["updated_wer"]) for row in group]
        mean_original = statistics.fmean(original_values)
        mean_updated = statistics.fmean(updated_values)
        std_updated = statistics.stdev(updated_values) if len(updated_values) > 1 else 0.0
        aggregate.append(
            {
                "dataset_tag": dataset_tag,
                "method": method,
                "epochs": epochs,
                "lr": lr,
                "n": str(len(group)),
                "original_wer_mean": f"{mean_original:.6f}",
                "updated_wer_mean": f"{mean_updated:.6f}",
                "updated_wer_std": f"{std_updated:.6f}",
                "absolute_delta_mean": f"{mean_updated - mean_original:.6f}",
                "relative_delta_pct_mean": f"{((mean_updated - mean_original) / mean_original) * 100:.2f}"
                if mean_original
                else "",
            }
        )
    return aggregate


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path, title: str, note: str) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    missing = [row for row in rows if row["status"] != "complete"]
    lines = [
        f"# {title}",
        "",
    ]
    if note:
        lines.extend([note, ""])
    lines.extend(
        [
            f"Completed cells: {complete}/{len(rows)}",
            "",
            "The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.",
            "",
            "## Aggregate",
            "",
            "| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_rows(rows):
        lines.append(
            "| {dataset_tag} | {method} | {epochs} | `{lr}` | {n} | {original_wer_mean} | "
            "{updated_wer_mean} | {updated_wer_std} | {absolute_delta_mean} | "
            "{relative_delta_pct_mean} |".format(**row)
        )
    if missing:
        lines.extend(["", "## Missing Cells", ""])
        for row in missing[:40]:
            lines.append(
                "- {dataset_tag} / {method} / repeat {repeat} / epoch {epochs} / lr `{lr}`".format(**row)
            )
        if len(missing) > 40:
            lines.append(f"- ... {len(missing) - 40} more missing cells")
    lines.extend(
        [
            "",
            "## Per Repeat",
            "",
            "| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset_tag} | {method} | {repeat} | {seed} | {epochs} | `{lr}` | {original_wer} | "
            "{updated_wer} | {absolute_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("exp/results/repro"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/results/repro/symphony/rob-108"))
    parser.add_argument("--datasets", default="tedlium earnings22 chime6 rev16 TAL")
    parser.add_argument("--methods", default="NoAug RFM RMM UFMR UVQLM")
    parser.add_argument("--repeats", default="1")
    parser.add_argument("--epoch1-lrs", default="1e-5 3e-5")
    parser.add_argument("--epoch5-lrs", default="1e-5")
    parser.add_argument("--csv-name", default="rob108_test_policy_evals.csv")
    parser.add_argument("--outcome-name", default="ROB-108_OUTCOME.md")
    parser.add_argument("--title", default="ROB-108 Test Policy Evaluations")
    parser.add_argument(
        "--note",
        default=(
            "Test split policy evals for RFM, RMM, UFMR, and UVQLM across TED-LIUM, "
            "Earnings22, CHiME-6, Rev16, and This American Life. NoAug rows are unadapted baselines."
        ),
    )
    args = parser.parse_args()

    rows = collect_rows(
        args.result_root,
        datasets=parse_strings(args.datasets),
        methods=parse_strings(args.methods),
        repeats=parse_ints(args.repeats),
        epoch1_lrs=parse_strings(args.epoch1_lrs),
        epoch5_lrs=parse_strings(args.epoch5_lrs),
    )
    write_csv(rows, args.output_dir / args.csv_name)
    write_markdown(rows, args.output_dir / args.outcome_name, args.title, args.note)
    print(f"Wrote {args.output_dir / args.outcome_name}")
    print(f"Wrote {args.output_dir / args.csv_name}")


if __name__ == "__main__":
    main()
