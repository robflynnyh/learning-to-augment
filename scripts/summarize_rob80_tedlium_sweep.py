#!/usr/bin/env python3
"""Summarize ROB-80 TED-LIUM policy LR sweep results."""

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

METHODS = ("RFM", "RMM", "UFMR")
EPOCHS = (1, 5)
LRS = ("5e-6", "1e-5", "2e-5")


def parse_result(path: Path, expected_epochs: int) -> tuple[float, float] | None:
    if not path.exists():
        return None

    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        match = RESULT_RE.search(line)
        if not match:
            continue
        if match.group("dataset") != "tedlium":
            continue
        if int(match.group("epochs")) != expected_epochs:
            continue
        return float(match.group("original")), float(match.group("updated"))
    return None


def collect_rows(result_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in METHODS:
        for epochs in EPOCHS:
            for lr in LRS:
                result_path = result_root / method / f"tedlium_epoch{epochs}_lr{lr}.txt"
                parsed = parse_result(result_path, epochs)
                row = {
                    "method": method,
                    "epochs": str(epochs),
                    "lr": lr,
                    "result_path": str(result_path),
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


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    lines = [
        "# ROB-80 TED-LIUM Policy LR Sweep",
        "",
        f"Completed cells: {complete}/{len(rows)}",
        "",
        "| Method | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {epochs} | `{lr}` | {original_wer} | {updated_wer} | "
            "{absolute_delta} | {relative_delta_pct} | {status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("exp/results/repro/sweeps"),
        help="Root containing RFM/RMM/UFMR result folders.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.result_root)
    write_csv(rows, args.result_root / "rob80_tedlium_policy_sweep.csv")
    write_markdown(rows, args.result_root / "ROB-80_OUTCOME.md")
    print(f"Wrote {args.result_root / 'ROB-80_OUTCOME.md'}")
    print(f"Wrote {args.result_root / 'rob80_tedlium_policy_sweep.csv'}")


if __name__ == "__main__":
    main()
