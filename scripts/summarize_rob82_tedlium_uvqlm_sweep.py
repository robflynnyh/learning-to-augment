#!/usr/bin/env python3
"""Summarize ROB-82 UVQLM TED-LIUM sweep results."""

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


def collect_rows(
    result_root: Path,
    method: str,
    dataset: str,
    split: str,
    tag_prefix: str,
    epochs: tuple[int, ...],
    lrs: tuple[str, ...],
    repeats: tuple[int, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for repeat in repeats:
        repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
        seed = 123456 + repeat - 1
        for epoch_count in epochs:
            for lr in lrs:
                result_path = result_root / method / f"{tag_prefix}_epoch{epoch_count}_lr{lr}{repeat_suffix}.txt"
                parsed = parse_result(result_path, dataset, split, epoch_count)
                row = {
                    "method": method,
                    "repeat": str(repeat),
                    "seed": str(seed),
                    "epochs": str(epoch_count),
                    "lr": lr,
                    "result_path": display_path(result_path),
                }
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
                else:
                    original, updated = parsed
                    row.update(
                        {
                            "status": "complete",
                            "original_wer": f"{original:.6f}",
                            "updated_wer": f"{updated:.6f}",
                            "absolute_delta": f"{updated - original:.6f}",
                            "relative_delta_pct": f"{((updated - original) / original) * 100:.2f}"
                            if original
                            else "",
                        }
                    )
                rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        key = (row["method"], row["epochs"], row["lr"])
        grouped.setdefault(key, []).append(row)

    aggregate: list[dict[str, str]] = []
    for (method, epochs, lr), group in grouped.items():
        original_values = [float(row["original_wer"]) for row in group]
        updated_values = [float(row["updated_wer"]) for row in group]
        mean_original = statistics.fmean(original_values)
        mean_updated = statistics.fmean(updated_values)
        std_updated = statistics.stdev(updated_values) if len(updated_values) > 1 else 0.0
        aggregate.append(
            {
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
    lines = [
        f"# {title}",
        "",
        f"Completed cells: {complete}/{len(rows)}",
        "",
        "## Aggregate",
        "",
        "| Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if note:
        lines[1:1] = ["", note]
    for row in aggregate_rows(rows):
        lines.append(
            "| {method} | {epochs} | `{lr}` | {n} | {original_wer_mean} | "
            "{updated_wer_mean} | {updated_wer_std} | {absolute_delta_mean} | "
            "{relative_delta_pct_mean} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per Repeat",
            "",
            "| Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {repeat} | {seed} | {epochs} | `{lr}` | {original_wer} | {updated_wer} | "
            "{absolute_delta} | {relative_delta_pct} | {status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--method", default="UVQLM")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--lrs", default="5e-6 1e-5 2e-5")
    parser.add_argument("--repeats", default="1 2")
    parser.add_argument("--csv-name", required=True)
    parser.add_argument("--outcome-name", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    rows = collect_rows(
        args.result_root,
        method=args.method,
        dataset=args.dataset,
        split=args.split,
        tag_prefix=args.tag_prefix,
        epochs=parse_ints(args.epochs),
        lrs=parse_strings(args.lrs),
        repeats=parse_ints(args.repeats),
    )
    csv_path = args.result_root / args.csv_name
    outcome_path = args.result_root / args.outcome_name
    write_csv(rows, csv_path)
    write_markdown(rows, outcome_path, args.title, args.note)
    print(f"Wrote {outcome_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
