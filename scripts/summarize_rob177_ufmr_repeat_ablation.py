#!/usr/bin/env python3
"""Summarize ROB-177 UFMR candidate-repeat ablation results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(
    r"Dataset:\s*(?P<dataset>\S+)\s*-\s*"
    r"Split:\s*(?P<split>\S+)\s*-\s*"
    r"Epochs:\s*(?P<epochs>\d+)\s*-\s*"
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*"
    r"Updated_WER:\s*(?P<updated>[0-9.]+)"
)


def parse_candidate_repeats(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


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


def result_row(
    *,
    candidate_repeats: int,
    source: str,
    result_path: Path,
    dataset_tag: str,
    dataset: str,
    split: str,
    epochs: int,
    lr: str,
) -> dict[str, str]:
    row = {
        "dataset_tag": dataset_tag,
        "dataset": dataset,
        "split": split,
        "method": "UFMR",
        "candidate_repeats": str(candidate_repeats),
        "source": source,
        "epochs": str(epochs),
        "lr": lr,
        "result_path": display_path(result_path),
    }
    parsed = parse_result(result_path, dataset, split, epochs)
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
        return row

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
    return row


def collect_rows(
    result_root: Path,
    candidate_repeats: tuple[int, ...],
    dataset_tag: str,
    dataset: str,
    split: str,
    epochs: int,
    lr: str,
    reference_result: Path | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for repeat_count in candidate_repeats:
        tag = f"{dataset_tag}_{split}_candidate_repeats{repeat_count}_epoch{epochs}_lr{lr}"
        rows.append(
            result_row(
                candidate_repeats=repeat_count,
                source="ROB-177 ablation",
                result_path=result_root / "UFMR" / f"{tag}.txt",
                dataset_tag=dataset_tag,
                dataset=dataset,
                split=split,
                epochs=epochs,
                lr=lr,
            )
        )

    if reference_result is not None:
        rows.append(
            result_row(
                candidate_repeats=15,
                source="ROB-108 default reference",
                result_path=reference_result,
                dataset_tag=dataset_tag,
                dataset=dataset,
                split=split,
                epochs=epochs,
                lr=lr,
            )
        )
    return sorted(rows, key=lambda row: (int(row["candidate_repeats"]), row["source"]))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    ablation_rows = [row for row in rows if row["source"] == "ROB-177 ablation"]
    missing = [row for row in ablation_rows if row["status"] != "complete"]
    lines = [
        "# ROB-177 UFMR Candidate-Repeat Ablation",
        "",
        "Earnings22 test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.",
        "",
        "All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.",
        "The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.",
        "",
        f"Completed rows: {complete}/{len(rows)}",
        f"Completed ROB-177 ablation rows: {len(ablation_rows) - len(missing)}/{len(ablation_rows)}",
        "",
        "| Candidate Repeats | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {candidate_repeats} | {source} | {original_wer} | {updated_wer} | "
            "{absolute_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )

    if missing:
        lines.extend(["", "## Missing ROB-177 Cells", ""])
        for row in missing:
            lines.append("- candidate repeats `{candidate_repeats}`".format(**row))

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("exp/results/repro/symphony/rob-177/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/results/repro/symphony/rob-177"))
    parser.add_argument("--candidate-repeats", default="2 5 10 20 40 100 200")
    parser.add_argument("--dataset-tag", default="earnings22")
    parser.add_argument("--dataset", default="earnings22")
    parser.add_argument("--split", default="test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--reference-result", type=Path, default=Path("exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt"))
    parser.add_argument("--csv-name", default="rob177_ufmr_repeat_ablation.csv")
    parser.add_argument("--outcome-name", default="ROB-177_OUTCOME.md")
    args = parser.parse_args()

    rows = collect_rows(
        args.result_root,
        candidate_repeats=parse_candidate_repeats(args.candidate_repeats),
        dataset_tag=args.dataset_tag,
        dataset=args.dataset,
        split=args.split,
        epochs=args.epochs,
        lr=args.lr,
        reference_result=args.reference_result,
    )
    write_csv(rows, args.output_dir / args.csv_name)
    write_markdown(rows, args.output_dir / args.outcome_name)
    print(f"Wrote {args.output_dir / args.outcome_name}")
    print(f"Wrote {args.output_dir / args.csv_name}")


if __name__ == "__main__":
    main()
