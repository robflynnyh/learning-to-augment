#!/usr/bin/env python3
"""Build the ROB-62 corrected RMM-vs-RFM comparison table."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("RFM", "RMM")
DATASETS = (
    ("tedlium", "TED-LIUM"),
    ("e22", "Earnings-22"),
    ("rev16", "Rev16"),
    ("chime6", "CHiME-6"),
    ("TAL", "TAL"),
)
RESULT_RE = re.compile(
    r"Original_WER:\s*(?P<original>[0-9.]+)\s*-\s*Updated_WER:\s*(?P<updated>[0-9.]+)"
)


def read_latest_result(path: Path) -> tuple[float | None, float | None, str]:
    if not path.exists():
        return None, None, "missing"
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    for line in reversed(lines):
        match = RESULT_RE.search(line)
        if match:
            return float(match["original"]), float(match["updated"]), "complete"
    return None, None, "unparseable"


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> int:
    rows: list[dict[str, str]] = []
    for tag, label in DATASETS:
        parsed = {
            method: read_latest_result(ROOT / method / f"{tag}.txt")
            for method in METHODS
        }
        rfm_original, rfm_updated, rfm_status = parsed["RFM"]
        rmm_original, rmm_updated, rmm_status = parsed["RMM"]
        rfm_abs_change = None
        rmm_abs_change = None
        delta = None
        if rfm_original is not None and rfm_updated is not None:
            rfm_abs_change = rfm_updated - rfm_original
        if rmm_original is not None and rmm_updated is not None:
            rmm_abs_change = rmm_updated - rmm_original
        if rfm_updated is not None and rmm_updated is not None:
            delta = rmm_updated - rfm_updated
        rows.append(
            {
                "dataset": label,
                "rfm_original_wer": fmt(rfm_original),
                "rfm_updated_wer": fmt(rfm_updated),
                "rfm_change": fmt(rfm_abs_change),
                "rmm_original_wer": fmt(rmm_original),
                "rmm_updated_wer": fmt(rmm_updated),
                "rmm_change": fmt(rmm_abs_change),
                "rmm_minus_rfm_updated_wer": fmt(delta),
                "rfm_status": rfm_status,
                "rmm_status": rmm_status,
            }
        )

    csv_path = ROOT / "comparison.csv"
    md_path = ROOT / "comparison.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0])
    with md_path.open("w") as handle:
        handle.write("# ROB-62 Corrected RMM vs RFM Comparison\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[key] for key in headers) + " |\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
