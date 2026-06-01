#!/usr/bin/env python3
"""Write ROB-158 large-ASR policy comparison CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("dataset_tag", "epochs", "lr", "repeat")
NUMERIC_FIELDS = ("original_wer", "updated_wer", "absolute_delta", "relative_delta_pct")


def read_complete_rows(path: Path, method: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if row["method"] != method or row["status"] != "complete":
            continue
        key = tuple(row[field] for field in KEY_FIELDS)
        complete[key] = row
    return complete


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cross_model_rows(
    small_rows: dict[tuple[str, str, str, str], dict[str, str]],
    large_rows: dict[tuple[str, str, str, str], dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(set(small_rows) & set(large_rows)):
        small = small_rows[key]
        large = large_rows[key]
        row = {
            "dataset_tag": key[0],
            "method": method,
            "epochs": key[1],
            "lr": key[2],
            "repeat": key[3],
        }
        for prefix, source in (("small_asr", small), ("large_asr", large)):
            for field in NUMERIC_FIELDS:
                row[f"{prefix}_{field}"] = source[field]
        large_rel = as_float(large, "relative_delta_pct")
        small_rel = as_float(small, "relative_delta_pct")
        row["large_minus_small_rel_delta_pct"] = f"{large_rel - small_rel:.2f}"
        row["large_minus_small_updated_wer"] = f"{as_float(large, 'updated_wer') - as_float(small, 'updated_wer'):.6f}"
        row["large_improves"] = str(as_float(large, "absolute_delta") < 0)
        row["large_rel_delta_stronger_than_small"] = str(large_rel < small_rel)
        rows.append(row)
    return rows


def same_model_rows(
    left_rows: dict[tuple[str, str, str, str], dict[str, str]],
    right_rows: dict[tuple[str, str, str, str], dict[str, str]],
    left_method: str,
    right_method: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(set(left_rows) & set(right_rows)):
        left = left_rows[key]
        right = right_rows[key]
        left_updated = as_float(left, "updated_wer")
        right_updated = as_float(right, "updated_wer")
        left_rel = as_float(left, "relative_delta_pct")
        right_rel = as_float(right, "relative_delta_pct")
        if left_updated < right_updated:
            updated_winner = left_method
        elif right_updated < left_updated:
            updated_winner = right_method
        else:
            updated_winner = "tie"
        if left_rel < right_rel:
            rel_winner = left_method
        elif right_rel < left_rel:
            rel_winner = right_method
        else:
            rel_winner = "tie"
        row = {
            "dataset_tag": key[0],
            "epochs": key[1],
            "lr": key[2],
            "repeat": key[3],
        }
        for prefix, source in ((left_method.lower(), left), (right_method.lower(), right)):
            for field in NUMERIC_FIELDS:
                row[f"{prefix}_{field}"] = source[field]
        row[f"{right_method.lower()}_minus_{left_method.lower()}_updated_wer"] = (
            f"{right_updated - left_updated:.6f}"
        )
        row[f"{right_method.lower()}_minus_{left_method.lower()}_rel_delta_pct"] = f"{right_rel - left_rel:.2f}"
        row["updated_wer_winner"] = updated_winner
        row["relative_delta_winner"] = rel_winner
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rob108-csv", type=Path, default=Path("exp/results/repro/symphony/rob-108/rob108_test_policy_evals.csv"))
    parser.add_argument("--rob158-rfm-csv", type=Path, default=Path("exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/rob158_rfm_large_asr_eval.csv"))
    parser.add_argument("--rob158-ufmr-csv", type=Path, default=Path("exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/rob158_ufmr_large_asr_eval.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048"))
    args = parser.parse_args()

    rob108_rfm = read_complete_rows(args.rob108_csv, "RFM")
    rob158_rfm = read_complete_rows(args.rob158_rfm_csv, "RFM")
    rob158_ufmr = read_complete_rows(args.rob158_ufmr_csv, "UFMR")

    rfm_cross_model = cross_model_rows(rob108_rfm, rob158_rfm, method="RFM")
    ufmr_vs_rfm = same_model_rows(rob158_ufmr, rob158_rfm, left_method="UFMR", right_method="RFM")

    rfm_cross_model_path = args.output_dir / "rob158_vs_rob108_rfm_comparison.csv"
    ufmr_vs_rfm_path = args.output_dir / "rob158_large_asr_ufmr_vs_rfm_comparison.csv"
    write_csv(rfm_cross_model_path, rfm_cross_model)
    write_csv(ufmr_vs_rfm_path, ufmr_vs_rfm)
    print(f"Wrote {rfm_cross_model_path} ({len(rfm_cross_model)} rows)")
    print(f"Wrote {ufmr_vs_rfm_path} ({len(ufmr_vs_rfm)} rows)")


if __name__ == "__main__":
    main()
