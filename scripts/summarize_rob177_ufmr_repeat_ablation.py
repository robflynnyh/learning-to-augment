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

DATASETS = {
    "earnings22": ("earnings22", "test"),
    "tedlium": ("tedlium", "test"),
}


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def parse_reference_results(raw: str | None, legacy_reference: Path | None) -> dict[str, Path]:
    if raw is None:
        refs = {
            "earnings22": Path("exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt"),
            "tedlium": Path("exp/results/repro/UFMR/tedlium_epoch1_lr1e-5.txt"),
        }
    else:
        refs = {}
        for item in raw.split():
            if "=" not in item:
                raise ValueError(f"Reference result spec must be dataset=path, got: {item}")
            dataset_tag, path = item.split("=", 1)
            refs[dataset_tag] = Path(path)

    if legacy_reference is not None:
        refs = {"earnings22": legacy_reference}
    return refs


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


def result_tag(
    *,
    dataset_tag: str,
    split: str,
    candidate_repeats: int,
    seed: int,
    first_seed: int | None,
    epochs: int,
    lr: str,
    legacy_first_seed_repeats: set[int],
) -> str:
    legacy_name = (
        dataset_tag == "earnings22"
        and seed == first_seed
        and candidate_repeats in legacy_first_seed_repeats
    )
    if legacy_name:
        return f"{dataset_tag}_{split}_candidate_repeats{candidate_repeats}_epoch{epochs}_lr{lr}"
    return f"{dataset_tag}_{split}_candidate_repeats{candidate_repeats}_seed{seed}_epoch{epochs}_lr{lr}"


def result_row(
    *,
    candidate_repeats: int,
    trial: int | None,
    seed: int | None,
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
        "trial": "" if trial is None else str(trial),
        "seed": "" if seed is None else str(seed),
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
    dataset_tags: tuple[str, ...],
    candidate_repeats: tuple[int, ...],
    seeds: tuple[int, ...],
    split: str,
    epochs: int,
    lr: str,
    reference_results: dict[str, Path],
    legacy_first_seed_repeats: set[int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    first_seed = seeds[0] if seeds else None
    for dataset_tag in dataset_tags:
        if dataset_tag not in DATASETS:
            raise ValueError(f"Unknown ROB-177 dataset tag: {dataset_tag}")
        dataset, default_split = DATASETS[dataset_tag]
        active_split = split or default_split
        for repeat_count in candidate_repeats:
            for trial_index, seed in enumerate(seeds, start=1):
                tag = result_tag(
                    dataset_tag=dataset_tag,
                    split=active_split,
                    candidate_repeats=repeat_count,
                    seed=seed,
                    first_seed=first_seed,
                    epochs=epochs,
                    lr=lr,
                    legacy_first_seed_repeats=legacy_first_seed_repeats,
                )
                rows.append(
                    result_row(
                        candidate_repeats=repeat_count,
                        trial=trial_index,
                        seed=seed,
                        source="ROB-177 ablation",
                        result_path=result_root / "UFMR" / f"{tag}.txt",
                        dataset_tag=dataset_tag,
                        dataset=dataset,
                        split=active_split,
                        epochs=epochs,
                        lr=lr,
                    )
                )

        reference_result = reference_results.get(dataset_tag)
        if reference_result is not None:
            rows.append(
                result_row(
                    candidate_repeats=15,
                    trial=None,
                    seed=None,
                    source="ROB-108 default reference",
                    result_path=reference_result,
                    dataset_tag=dataset_tag,
                    dataset=dataset,
                    split=active_split,
                    epochs=epochs,
                    lr=lr,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset_tag"],
            int(row["candidate_repeats"]),
            row["source"],
            int(row["trial"]) if row["trial"] else 0,
        ),
    )


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_dataset_repeat(rows: list[dict[str, str]]) -> list[tuple[str, str, list[float]]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["source"] != "ROB-177 ablation" or row["status"] != "complete":
            continue
        key = (row["dataset_tag"], row["candidate_repeats"])
        grouped.setdefault(key, []).append(float(row["updated_wer"]))
    return [(dataset_tag, repeat_count, values) for (dataset_tag, repeat_count), values in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1])))]


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    ablation_rows = [row for row in rows if row["source"] == "ROB-177 ablation"]
    missing = [row for row in ablation_rows if row["status"] != "complete"]
    dataset_list = ", ".join(f"`{dataset}`" for dataset in sorted({row["dataset_tag"] for row in ablation_rows}))
    repeat_list = ", ".join(f"`{repeat}`" for repeat in sorted({int(row["candidate_repeats"]) for row in ablation_rows}))
    lines = [
        "# ROB-177 UFMR Candidate-Repeat Ablation",
        "",
        f"{dataset_list} test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.",
        "",
        "All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.",
        "The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.",
        f"Candidate-repeat settings: {repeat_list}.",
        "Each ROB-177 candidate-repeat setting has three seed trials when complete.",
        "",
        f"Completed rows: {complete}/{len(rows)}",
        f"Completed ROB-177 ablation rows: {len(ablation_rows) - len(missing)}/{len(ablation_rows)}",
        "",
        "## Per-Trial Results",
        "",
        "| Dataset | Candidate Repeats | Trial | Seed | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset_tag} | {candidate_repeats} | {trial} | {seed} | {source} | {original_wer} | {updated_wer} | "
            "{absolute_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )

    complete_ablation = [row for row in ablation_rows if row["status"] == "complete"]
    if complete_ablation:
        lines.extend(["", "## Candidate-Repeat Summary", ""])
        lines.append("| Dataset | Candidate Repeats | Complete Trials | Mean Updated WER | Std Updated WER | Best Updated WER | Worst Updated WER |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for dataset_tag, repeat_count, values in aggregate_by_dataset_repeat(rows):
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance ** 0.5
            lines.append(
                f"| {dataset_tag} | {repeat_count} | {len(values)} | {mean:.6f} | {std:.6f} | {min(values):.6f} | {max(values):.6f} |"
            )

    if missing:
        lines.extend(["", "## Missing ROB-177 Cells", ""])
        for row in missing:
            lines.append(
                "- {dataset_tag}: candidate repeats `{candidate_repeats}`, trial `{trial}`, seed `{seed}`".format(**row)
            )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("exp/results/repro/symphony/rob-177/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/results/repro/symphony/rob-177"))
    parser.add_argument("--candidate-repeats", default="1 2 5 10 15 20 40 100 200 1000")
    parser.add_argument("--seeds", default="123456 123457 123458")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--reference-results", default=None)
    parser.add_argument("--reference-result", type=Path, default=None)
    parser.add_argument("--legacy-first-seed-repeats", default="2 5 10 20 40 100 200")
    parser.add_argument("--legacy-first-seed", default=None, choices=("0", "1"))
    parser.add_argument("--csv-name", default="rob177_ufmr_repeat_ablation.csv")
    parser.add_argument("--outcome-name", default="ROB-177_OUTCOME.md")
    args = parser.parse_args()

    if args.datasets is not None:
        dataset_tags = parse_strings(args.datasets)
    elif args.dataset_tag is not None:
        dataset_tags = (args.dataset_tag,)
    else:
        dataset_tags = ("earnings22", "tedlium")

    if args.legacy_first_seed == "0":
        legacy_first_seed_repeats: set[int] = set()
    else:
        legacy_first_seed_repeats = set(parse_ints(args.legacy_first_seed_repeats))

    rows = collect_rows(
        args.result_root,
        dataset_tags=dataset_tags,
        candidate_repeats=parse_ints(args.candidate_repeats),
        seeds=parse_ints(args.seeds),
        split=args.split,
        epochs=args.epochs,
        lr=args.lr,
        reference_results=parse_reference_results(args.reference_results, args.reference_result),
        legacy_first_seed_repeats=legacy_first_seed_repeats,
    )
    write_csv(rows, args.output_dir / args.csv_name)
    write_markdown(rows, args.output_dir / args.outcome_name)
    print(f"Wrote {args.output_dir / args.outcome_name}")
    print(f"Wrote {args.output_dir / args.csv_name}")


if __name__ == "__main__":
    main()
