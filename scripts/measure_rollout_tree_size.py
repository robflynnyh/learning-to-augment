#!/usr/bin/env python3
"""Measure exact file-tree sizes for a rollout directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


BLOCK_SIZE = 512


@dataclass
class Bucket:
    files: int = 0
    directories: int = 0
    symlinks: int = 0
    other: int = 0
    apparent_bytes: int = 0
    allocated_bytes: int = 0

    def add_stat(self, mode: int, size: int, blocks: int) -> None:
        if stat.S_ISREG(mode):
            self.files += 1
        elif stat.S_ISDIR(mode):
            self.directories += 1
        elif stat.S_ISLNK(mode):
            self.symlinks += 1
        else:
            self.other += 1
        self.apparent_bytes += size
        self.allocated_bytes += blocks * BLOCK_SIZE


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}" if unit != "B" else f"{value} B"
        number /= 1024.0
    return f"{value} B"


def top_level_bucket(root: Path, path: Path) -> str:
    if path == root:
        return "."
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else "."


def iter_tree(root: Path) -> Iterable[Tuple[Path, os.stat_result]]:
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        current_stat = current.lstat()
        yield current, current_stat
        if not stat.S_ISDIR(current_stat.st_mode):
            continue
        try:
            entries = list(os.scandir(current))
        except OSError as error:
            raise RuntimeError(f"Could not scan directory {current}: {error}") from error
        for entry in reversed(entries):
            stack.append(Path(entry.path))


def measure(root: Path, progress_every: int) -> Dict[str, object]:
    total = Bucket()
    top_level: DefaultDict[str, Bucket] = defaultdict(Bucket)
    suffixes: DefaultDict[str, Bucket] = defaultdict(Bucket)
    scanned = 0

    for path, st in iter_tree(root):
        scanned += 1
        mode = st.st_mode
        total.add_stat(mode, st.st_size, getattr(st, "st_blocks", 0))
        top_level[top_level_bucket(root, path)].add_stat(mode, st.st_size, getattr(st, "st_blocks", 0))
        if stat.S_ISREG(mode):
            suffix = path.suffix or "<no suffix>"
            suffixes[suffix].add_stat(mode, st.st_size, getattr(st, "st_blocks", 0))
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                f"[measure-rollout-size] scanned={scanned} "
                f"files={total.files} apparent={human_bytes(total.apparent_bytes)} "
                f"allocated={human_bytes(total.allocated_bytes)}",
                flush=True,
            )

    return {
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": asdict(total),
        "top_level": {name: asdict(bucket) for name, bucket in sorted(top_level.items())},
        "suffixes": {name: asdict(bucket) for name, bucket in sorted(suffixes.items())},
    }


def write_csv(path: Path, rows: Dict[str, Dict[str, int]]) -> None:
    fieldnames = [
        "name",
        "files",
        "directories",
        "symlinks",
        "other",
        "apparent_bytes",
        "allocated_bytes",
        "apparent_human",
        "allocated_human",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in rows.items():
            writer.writerow(
                {
                    **{"name": name},
                    **row,
                    "apparent_human": human_bytes(row["apparent_bytes"]),
                    "allocated_human": human_bytes(row["allocated_bytes"]),
                }
            )


def write_outcome(path: Path, data: Dict[str, object], command: str) -> None:
    total = data["total"]
    assert isinstance(total, dict)
    top_level = data["top_level"]
    assert isinstance(top_level, dict)

    lines = [
        "# ROB-106 UVQLM Rollout Folder Size",
        "",
        f"- Root: `{data['root']}`",
        f"- Generated UTC: `{data['generated_at_utc']}`",
        f"- Command: `{command}`",
        "",
        "## Total",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Files | {total['files']} |",
        f"| Directories | {total['directories']} |",
        f"| Symlinks | {total['symlinks']} |",
        f"| Other entries | {total['other']} |",
        f"| Apparent bytes | {total['apparent_bytes']} ({human_bytes(int(total['apparent_bytes']))}) |",
        f"| Allocated bytes | {total['allocated_bytes']} ({human_bytes(int(total['allocated_bytes']))}) |",
        "",
        "## Top-Level Breakdown",
        "",
        "| Name | Files | Directories | Apparent bytes | Allocated bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row_obj in top_level.items():
        row = dict(row_obj)
        lines.append(
            f"| `{name}` | {row['files']} | {row['directories']} | "
            f"{row['apparent_bytes']} ({human_bytes(int(row['apparent_bytes']))}) | "
            f"{row['allocated_bytes']} ({human_bytes(int(row['allocated_bytes']))}) |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Rollout directory to measure")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV/OUTCOME outputs")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress after this many filesystem entries; use 0 to disable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not root.exists():
        raise SystemExit(f"Rollout root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Rollout root is not a directory: {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"python3 scripts/measure_rollout_tree_size.py --root {root} "
        f"--output-dir {output_dir} --progress-every {args.progress_every}"
    )
    data = measure(root, args.progress_every)
    (output_dir / "uvqlm_rollout_size.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    write_csv(output_dir / "uvqlm_rollout_size_by_top_level.csv", data["top_level"])
    write_csv(output_dir / "uvqlm_rollout_size_by_suffix.csv", data["suffixes"])
    write_outcome(output_dir / "OUTCOME.md", data, command)

    total = data["total"]
    assert isinstance(total, dict)
    print("[measure-rollout-size] complete")
    print(f"[measure-rollout-size] root={root}")
    print(f"[measure-rollout-size] files={total['files']} directories={total['directories']}")
    print(
        f"[measure-rollout-size] apparent_bytes={total['apparent_bytes']} "
        f"({human_bytes(int(total['apparent_bytes']))})"
    )
    print(
        f"[measure-rollout-size] allocated_bytes={total['allocated_bytes']} "
        f"({human_bytes(int(total['allocated_bytes']))})"
    )
    print(f"[measure-rollout-size] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
