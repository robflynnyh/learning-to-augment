#!/usr/bin/env python3
"""Validate ROB-338 completeness, effective seeds, and corrected-run logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--not-before", required=True)
    return parser.parse_args()


def parse_timestamp(value: str) -> float:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def augmentation_seed(config_path: Path) -> str | None:
    in_augmentation_config = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "augmentation_config:":
            in_augmentation_config = True
            continue
        if in_augmentation_config and line and not line.startswith(" "):
            in_augmentation_config = False
        if in_augmentation_config and stripped.startswith("seed:"):
            return stripped.split(":", 1)[1].strip()
    return None


def reward_token(reward: str) -> str:
    value = float(reward)
    if value == 0.0:
        return "0"
    if value == 1.0:
        return "1"
    return reward.replace(".", "p").replace("-", "m")


def successful_logs(row: dict[str, str], log_dir: Path, not_before: float) -> list[Path]:
    tag = (
        f"{row['dataset_tag']}_{row['split']}_epoch{row['epochs']}_"
        f"lr{row['lr']}_repeat{row['repeat']}"
    )
    pattern = f"{tag}-reward{reward_token(row['reward'])}-*.log"
    expected_lines = (
        f"[rob338-stanage-cell] dataset={row['dataset_tag']}",
        f"[rob338-stanage-cell] reward={row['reward']}",
        f"[rob338-stanage-cell] epoch={row['epochs']}",
        f"[rob338-stanage-cell] repeat={row['repeat']}",
        f"[rob338-stanage-cell] seed={row['seed']}",
        "[rob338-stanage-cell] finished",
    )
    matches = []
    for path in sorted(log_dir.glob(pattern)):
        if path.stat().st_mtime < not_before:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if all(line in text for line in expected_lines):
            matches.append(path)
    return matches


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    not_before = parse_timestamp(args.not_before)
    log_dir = args.result_root / "logs" / "stanage" / "rob338"

    failures: list[str] = []
    complete = [row for row in rows if row["status"] == "complete"]
    print(f"[rob338-provenance] complete={len(complete)}/{len(rows)}")
    if len(rows) != 60 or len(complete) != 60:
        failures.append(f"expected 60 complete rows, got {len(complete)}/{len(rows)}")

    counts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in complete:
        counts[(row["reward"], row["dataset_tag"], row["epochs"])].add(row["repeat"])
    for key, repeats in sorted(counts.items()):
        if repeats != {"1", "2", "3"}:
            failures.append(f"incomplete aggregate {key}: repeats={sorted(repeats)}")
    if len(counts) != 20:
        failures.append(f"expected 20 aggregates, got {len(counts)}")

    proven_new_cells = 0
    for row in rows:
        config_path = Path(row["config_path"])
        observed_seed = augmentation_seed(config_path)
        if observed_seed != row["seed"]:
            failures.append(
                "seed mismatch "
                f"reward={row['reward']} dataset={row['dataset_tag']} epoch={row['epochs']} "
                f"repeat={row['repeat']} expected={row['seed']} observed={observed_seed}"
            )

        if row["repeat"] == "1":
            continue
        result_path = Path(row["result_path"])
        if not result_path.exists() or result_path.stat().st_mtime < not_before:
            failures.append(
                "stale result "
                f"reward={row['reward']} dataset={row['dataset_tag']} epoch={row['epochs']} "
                f"repeat={row['repeat']} path={result_path}"
            )
            continue
        logs = successful_logs(row, log_dir, not_before)
        if not logs:
            failures.append(
                "missing corrected-run log "
                f"reward={row['reward']} dataset={row['dataset_tag']} epoch={row['epochs']} "
                f"repeat={row['repeat']}"
            )
            continue
        proven_new_cells += 1

    print(f"[rob338-provenance] corrected_run_logs={proven_new_cells}/40")
    if failures:
        for failure in failures:
            print(f"[rob338-provenance] failure: {failure}")
        return 1
    print("[rob338-provenance] validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
