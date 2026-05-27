#!/usr/bin/env python3
"""Summarize ROB-132 audio SSL-conditioned fixed-reward self-training evals."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(
    r"ID: (?P<id>.*?) - Dataset: (?P<dataset>.*?) - Split: (?P<split>.*?) "
    r"- Epochs: (?P<epochs>\d+) - Original_WER: (?P<original>[0-9.eE+-]+) "
    r"- Updated_WER: (?P<updated>[0-9.eE+-]+)"
)


def parse_result(path: Path, expected_dataset: str, expected_split: str, expected_epochs: int) -> dict[str, float] | None:
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
        "original_wer": original,
        "updated_wer": updated,
        "wer_delta": updated - original,
        "relative_delta_pct": ((updated - original) / original) * 100.0 if original else 0.0,
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split())


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def reward_tag(reward: float) -> str:
    return str(reward).replace(".", "p").replace("-", "m")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def expected_rows(
    result_root: Path,
    dataset: str,
    split: str,
    rewards: tuple[float, ...],
    epochs: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for reward in rewards:
        method = f"AudioRewardConditionedMaskLMReward{reward_tag(reward)}"
        for epoch_count in epochs:
            tag = f"{dataset}_{split}_reward{reward_tag(reward)}_epoch{epoch_count}_lr{lr}"
            rows.append(
                {
                    "condition": f"fixed_reward_{reward:g}",
                    "conditioning_reward": f"{reward:g}",
                    "dataset": dataset,
                    "split": split,
                    "method": method,
                    "epochs": str(epoch_count),
                    "lr": lr,
                    "config_path": display_path(result_root / method / "configs" / f"{tag}.yaml"),
                    "result_path": display_path(result_root / method / f"{tag}.txt"),
                }
            )
    return rows


def collect_rows(
    result_root: Path,
    dataset: str,
    split: str,
    rewards: tuple[float, ...],
    epochs: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows = expected_rows(result_root, dataset, split, rewards, epochs, lr)
    for row in rows:
        parsed = parse_result(
            Path(row["result_path"]),
            expected_dataset=dataset,
            expected_split=split,
            expected_epochs=int(row["epochs"]),
        )
        if parsed is None:
            row.update(
                {
                    "status": "missing",
                    "original_wer": "",
                    "updated_wer": "",
                    "wer_delta": "",
                    "relative_delta_pct": "",
                }
            )
            continue
        row.update(
            {
                "status": "complete",
                "original_wer": fmt(parsed["original_wer"]),
                "updated_wer": fmt(parsed["updated_wer"]),
                "wer_delta": fmt(parsed["wer_delta"]),
                "relative_delta_pct": f"{parsed['relative_delta_pct']:.2f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    checkpoint: str,
    command: str,
    branch: str,
    commit: str,
    log_path: str,
    screen_log_path: str,
) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    csv_path = path.with_name("rob132_audio_ssl_self_train_fixed_rewards.csv")
    lines = [
        "# ROB-132 Audio SSL Self-Training Fixed-Reward Eval",
        "",
        "## Metadata",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder",
        "- Reward control: fixed scalar `conditioning_reward`, no `conditioning_reward_range`",
        "- Dataset: `tedlium`, split `dev`",
        "- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Main log: `{log_path}`",
        f"- Screen log: `{screen_log_path}`",
        f"- Queued command: `{command}`",
        "",
        f"Completed cells: `{complete}/{len(rows)}`.",
        "",
        "## Results",
        "",
        "| Reward | Epochs | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {conditioning_reward} | {epochs} | {original_wer} | {updated_wer} | "
            "{wer_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )
    missing = [row for row in rows if row["status"] != "complete"]
    if missing:
        lines.extend(["", "## Missing Cells", ""])
        for row in missing:
            lines.append("- reward `{conditioning_reward}`, epoch `{epochs}`".format(**row))
    lines.extend(["", "CSV artifact:", "", f"```text\n{csv_path}\n```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/"
            "rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1"
        ),
    )
    parser.add_argument("--dataset", default="tedlium")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--rewards", default="1.0 0.0")
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--screen-log-path", required=True)
    parser.add_argument("--csv-name", default="rob132_audio_ssl_self_train_fixed_rewards.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    rows = collect_rows(
        args.result_root,
        dataset=args.dataset,
        split=args.split,
        rewards=parse_floats(args.rewards),
        epochs=parse_ints(args.epochs),
        lr=args.lr,
    )
    write_csv(rows, args.result_root / args.csv_name)
    write_markdown(
        rows,
        args.result_root / args.outcome_name,
        checkpoint=args.checkpoint,
        command=args.command,
        branch=args.branch,
        commit=args.commit,
        log_path=args.log_path,
        screen_log_path=args.screen_log_path,
    )
    print(f"[rob132-selftrain-summary] wrote {args.result_root / args.csv_name}")
    print(f"[rob132-selftrain-summary] wrote {args.result_root / args.outcome_name}")
    print(f"[rob132-selftrain-summary] completed {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} cells")


if __name__ == "__main__":
    main()
