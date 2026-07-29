#!/usr/bin/env python3
"""Summarize ROB-337 learnt-augmentation repeat evals."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


RESULT_RE = re.compile(
    r"ID: (?P<id>.*?) - Dataset: (?P<dataset>.*?) - Split: (?P<split>.*?) "
    r"- Epochs: (?P<epochs>\d+) - Original_WER: (?P<original>[0-9.eE+-]+) "
    r"- Updated_WER: (?P<updated>[0-9.eE+-]+)"
)

METHOD_LABELS = {
    "UFMR": "UFMR",
    "RFM": "RFM",
    "RMM": "RMM",
    "UVQLM": "UC-MLM",
}

DATASETS = {
    "tedlium": ("tedlium", "test", "TED-LIUM"),
    "earnings22": ("earnings22", "test", "Earnings-22"),
    "rev16": ("rev16", "test", "Rev16"),
    "TAL": ("this_american_life", "test", "TAL"),
    "tal": ("this_american_life", "test", "TAL"),
    "this_american_life": ("this_american_life", "test", "TAL"),
    "chime6": ("chime6", "test", "CHiME-6"),
}


def parse_result(path: Path, expected_dataset: str, expected_split: str, expected_epochs: int) -> dict[str, str] | None:
    if not path.exists():
        return None
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        if match.group("dataset") != expected_dataset:
            continue
        if match.group("split") != expected_split:
            continue
        if int(match.group("epochs")) != expected_epochs:
            continue
        matches.append(match)
    if not matches:
        return None
    match = matches[-1]
    original = float(match.group("original"))
    updated = float(match.group("updated"))
    return {
        "result_id": match.group("id"),
        "original_wer": f"{original:.9f}",
        "updated_wer": f"{updated:.9f}",
        "wer_delta": f"{updated - original:.9f}",
        "relative_delta_pct": f"{((updated - original) / original) * 100.0:.2f}" if original else "",
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def seed_for_repeat(repeat: int) -> int:
    if repeat == 1:
        return 123456
    return repeat * 100000 + 23456


def paths_for_repeat(
    base_repro_dir: Path,
    result_root: Path,
    method: str,
    dataset_tag: str,
    epochs: int,
    lr: str,
    repeat: int,
) -> tuple[Path, Path]:
    if repeat == 1:
        base_tag = f"{dataset_tag}_epoch{epochs}_lr{lr}"
        return (
            base_repro_dir / method / "configs" / f"{base_tag}.yaml",
            base_repro_dir / method / f"{base_tag}.txt",
        )
    repeat_tag = f"{dataset_tag}_epoch{epochs}_lr{lr}_repeat{repeat}"
    return (
        result_root / method / "configs" / f"{repeat_tag}.yaml",
        result_root / method / f"{repeat_tag}.txt",
    )


def expected_rows(
    base_repro_dir: Path,
    result_root: Path,
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    repeats: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in methods:
        for dataset_tag in datasets:
            dataset, split, dataset_label = DATASETS[dataset_tag]
            for epoch_count in epochs:
                for repeat in repeats:
                    config_path, result_path = paths_for_repeat(
                        base_repro_dir, result_root, method, dataset_tag, epoch_count, lr, repeat
                    )
                    rows.append(
                        {
                            "method": method,
                            "table_method": method_label(method),
                            "dataset_tag": dataset_tag,
                            "dataset": dataset,
                            "dataset_label": dataset_label,
                            "split": split,
                            "epochs": str(epoch_count),
                            "lr": lr,
                            "repeat": str(repeat),
                            "seed": str(seed_for_repeat(repeat)),
                            "config_path": display_path(config_path),
                            "result_path": display_path(result_path),
                        }
                    )
    return rows


def collect_rows(
    base_repro_dir: Path,
    result_root: Path,
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    repeats: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows = expected_rows(base_repro_dir, result_root, methods, datasets, epochs, repeats, lr)
    for row in rows:
        parsed = parse_result(Path(row["result_path"]), row["dataset"], row["split"], int(row["epochs"]))
        if parsed is None:
            row.update(
                {
                    "status": "missing",
                    "result_id": "",
                    "original_wer": "",
                    "updated_wer": "",
                    "wer_delta": "",
                    "relative_delta_pct": "",
                }
            )
            continue
        row.update({"status": "complete", **parsed})
    return rows


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        grouped.setdefault((row["method"], row["dataset_tag"], row["epochs"]), []).append(row)

    aggregate: list[dict[str, str]] = []
    for (method, dataset_tag, epochs), group in sorted(grouped.items(), key=lambda item: item[0]):
        original_values = [float(row["original_wer"]) for row in group]
        updated_values = [float(row["updated_wer"]) for row in group]
        mean_original = statistics.fmean(original_values)
        mean_updated = statistics.fmean(updated_values)
        std_updated = statistics.stdev(updated_values) if len(updated_values) > 1 else 0.0
        aggregate.append(
            {
                "method": method,
                "table_method": method_label(method),
                "dataset_tag": dataset_tag,
                "dataset": DATASETS[dataset_tag][0],
                "dataset_label": DATASETS[dataset_tag][2],
                "epochs": epochs,
                "lr": group[0]["lr"],
                "n": str(len(group)),
                "repeats": ",".join(row["repeat"] for row in sorted(group, key=lambda item: int(item["repeat"]))),
                "seeds": ",".join(row["seed"] for row in sorted(group, key=lambda item: int(item["repeat"]))),
                "original_wer_mean": f"{mean_original:.9f}",
                "updated_wer_mean": f"{mean_updated:.9f}",
                "updated_wer_std": f"{std_updated:.9f}",
                "absolute_delta_mean": f"{mean_updated - mean_original:.9f}",
                "relative_delta_pct_mean": f"{((mean_updated - mean_original) / mean_original) * 100.0:.2f}"
                if mean_original
                else "",
            }
        )
    return aggregate


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    per_repeat_csv: Path,
    aggregate_csv: Path,
    command: str,
    branch: str,
    commit: str,
    log_path: str,
    queued_jobs_path: str,
) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    missing = [row for row in rows if row["status"] != "complete"]
    aggregates = aggregate_rows(rows)
    incomplete_aggregates = [
        row for row in aggregates if int(row["n"]) != 3
    ]
    expected_groups = {
        (row["method"], row["dataset_tag"], row["epochs"])
        for row in rows
    }
    complete_groups = {
        (row["method"], row["dataset_tag"], row["epochs"])
        for row in aggregates
        if int(row["n"]) == 3
    }
    absent_groups = sorted(expected_groups - complete_groups)

    lines = [
        "# ROB-337 Learnt Augmentation Repeat Eval",
        "",
        "## Metadata",
        "",
        "- Scope: UFMR, RFM, RMM, and UC-MLM/UVQLM test-set learnt augmentation WER cells.",
        "- Datasets: TED-LIUM, Earnings-22, Rev16, TAL, CHiME-6; all `test` split.",
        "- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, 6L/2048 ASR.",
        "- Repeat policy: repeat 1 is the existing ROB-108 artifact; repeats 2 and 3 are ROB-337 Stanage jobs.",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Finalizer log: `{log_path}`",
        f"- Queued jobs: `{queued_jobs_path}`",
        f"- Queued command: `{command}`",
        "",
        f"Completed per-repeat rows: `{complete}/{len(rows)}`.",
        "",
        "## Aggregate",
        "",
        "| Method | Dataset | Epochs | N | Repeats | Seeds | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Rel Delta % |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {table_method} | {dataset_label} | {epochs} | {n} | {repeats} | {seeds} | "
            "{original_wer_mean} | {updated_wer_mean} | {updated_wer_std} | {relative_delta_pct_mean} |".format(**row)
        )
    if missing or incomplete_aggregates or absent_groups:
        lines.extend(["", "## Missing Or Incomplete Cells", ""])
        for row in missing:
            lines.append(
                "- {table_method} / {dataset_label} / epoch {epochs} / repeat {repeat} / seed {seed}: "
                "`{result_path}`".format(**row)
            )
        for method, dataset_tag, epochs in absent_groups:
            lines.append(f"- Aggregate incomplete: {method_label(method)} / {DATASETS[dataset_tag][2]} / epoch {epochs}")
    else:
        lines.extend(["", "## Missing Or Incomplete Cells", "", "None. All affected cells have `N=3`."])

    lines.extend(
        [
            "",
            "## Per Repeat",
            "",
            "| Method | Dataset | Epochs | Repeat | Seed | Original WER | Updated WER | Status | Result |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {table_method} | {dataset_label} | {epochs} | {repeat} | {seed} | {original_wer} | "
            "{updated_wer} | {status} | `{result_path}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "CSV artifacts:",
            "",
            f"```text\n{per_repeat_csv}\n{aggregate_csv}\n```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repro-dir", type=Path, default=Path("exp/results/repro"))
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("exp/results/repro/learnt_augmentation_repeats/rob337"),
    )
    parser.add_argument("--methods", default="UFMR RFM RMM UVQLM")
    parser.add_argument("--datasets", default="tedlium earnings22 rev16 TAL chime6")
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--repeats", default="1 2 3")
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--queued-jobs-path", required=True)
    parser.add_argument("--per-repeat-csv-name", default="rob337_learnt_augmentation_repeats.csv")
    parser.add_argument("--aggregate-csv-name", default="rob337_learnt_augmentation_repeats_aggregate.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    rows = collect_rows(
        args.base_repro_dir,
        args.result_root,
        methods=parse_strings(args.methods),
        datasets=parse_strings(args.datasets),
        epochs=parse_ints(args.epochs),
        repeats=parse_ints(args.repeats),
        lr=args.lr,
    )
    aggregate = aggregate_rows(rows)
    per_repeat_csv = args.result_root / args.per_repeat_csv_name
    aggregate_csv = args.result_root / args.aggregate_csv_name
    write_csv(rows, per_repeat_csv)
    write_csv(aggregate, aggregate_csv)
    write_markdown(
        rows,
        args.result_root / args.outcome_name,
        per_repeat_csv=per_repeat_csv,
        aggregate_csv=aggregate_csv,
        command=args.command,
        branch=args.branch,
        commit=args.commit,
        log_path=args.log_path,
        queued_jobs_path=args.queued_jobs_path,
    )
    print(f"[rob337-summary] wrote {per_repeat_csv}")
    print(f"[rob337-summary] wrote {aggregate_csv}")
    print(f"[rob337-summary] wrote {args.result_root / args.outcome_name}")
    print(f"[rob337-summary] completed {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} repeat rows")
    full_n = sum(int(row["n"]) == 3 for row in aggregate)
    print(f"[rob337-summary] aggregate_n3={full_n}/{len(aggregate)}")


if __name__ == "__main__":
    main()
