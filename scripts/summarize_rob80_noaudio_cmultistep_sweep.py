#!/usr/bin/env python3
"""Summarize ROB-80 no-audio CMultiStepVQLM TED-LIUM sweep results."""

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
    methods: tuple[str, ...],
    dataset: str,
    split: str,
    tag_prefix: str,
    epochs: tuple[int, ...],
    lrs: tuple[str, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in methods:
        for epoch_count in epochs:
            for lr in lrs:
                result_path = result_root / method / f"{tag_prefix}_epoch{epoch_count}_lr{lr}.txt"
                parsed = parse_result(result_path, dataset, split, epoch_count)
                row = {
                    "method": method,
                    "conditioning": conditioning_label(method),
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


def conditioning_label(method: str) -> str:
    if method == "CMultiStepVQLM":
        return "fixed_1.0"
    if method == "CMultiStepVQLMRandomReward":
        return "uniform_0.5_1.0"
    return method


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
        "| Method | Conditioning | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if note:
        lines[1:1] = ["", note]
    for row in rows:
        lines.append(
            "| {method} | {conditioning} | {epochs} | `{lr}` | {original_wer} | {updated_wer} | "
            "{absolute_delta} | {relative_delta_pct} | {status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_epochs(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def parse_lrs(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def parse_methods(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("exp/results/repro/sweeps/no_audio_cmultistep_vqlm"),
        help="Root containing no-audio CMultiStepVQLM result folders.",
    )
    parser.add_argument("--method", default=None, help="Single method to summarize; kept for compatibility.")
    parser.add_argument("--methods", default="CMultiStepVQLM", help="Space-separated method folders to summarize.")
    parser.add_argument("--dataset", default="tedlium")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--tag-prefix", default="tedlium_dev")
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--lrs", default="5e-6 1e-5 2e-5")
    parser.add_argument("--csv-name", default="rob80_tedlium_noaudio_cmultistep_sweep.csv")
    parser.add_argument("--outcome-name", default="ROB-80_NOAUDIO_CMULTISTEP_OUTCOME.md")
    parser.add_argument("--title", default="ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM LR Sweep")
    parser.add_argument(
        "--note",
        default=(
            "No-audio CMultiStepVQLM uses `ConditionalMultiStepMaskGenerator` "
            "with `condition_on_audio: false`, so generation is conditioned on "
            "reward/signal inputs and recurrent mask history, not raw audio."
        ),
    )
    args = parser.parse_args()

    rows = collect_rows(
        args.result_root,
        methods=parse_methods(args.method or args.methods),
        dataset=args.dataset,
        split=args.split,
        tag_prefix=args.tag_prefix,
        epochs=parse_epochs(args.epochs),
        lrs=parse_lrs(args.lrs),
    )
    csv_path = args.result_root / args.csv_name
    outcome_path = args.result_root / args.outcome_name
    write_csv(rows, csv_path)
    write_markdown(rows, outcome_path, args.title, args.note)
    print(f"Wrote {outcome_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
