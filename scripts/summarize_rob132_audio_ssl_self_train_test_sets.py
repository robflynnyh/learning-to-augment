#!/usr/bin/env python3
"""Summarize ROB-132 audio SSL fixed-reward test-set self-training evals."""

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

DATASETS = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
}


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


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def reward_tag(reward: str) -> str:
    return reward.replace(".", "p").replace("-", "m")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def expected_rows(
    result_root: Path,
    rewards: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for reward in rewards:
        method = f"AudioRewardConditionedMaskLMReward{reward_tag(reward)}"
        for dataset_tag in datasets:
            dataset, split = DATASETS[dataset_tag]
            for epoch_count in epochs:
                tag = f"{dataset_tag}_{split}_reward{reward_tag(reward)}_epoch{epoch_count}_lr{lr}"
                rows.append(
                    {
                        "condition": f"fixed_reward_{reward}",
                        "label": f"fixed conditioning reward {reward}",
                        "reward": reward,
                        "dataset_tag": dataset_tag,
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
    rewards: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    lr: str,
) -> list[dict[str, str]]:
    rows = expected_rows(result_root, rewards, datasets, epochs, lr)
    for row in rows:
        parsed = parse_result(
            Path(row["result_path"]),
            expected_dataset=row["dataset"],
            expected_split=row["split"],
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


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        grouped.setdefault((row["reward"], row["dataset_tag"], row["epochs"]), []).append(row)

    aggregate: list[dict[str, str]] = []
    for (reward, dataset_tag, epochs), group in sorted(
        grouped.items(), key=lambda item: (float(item[0][0]), item[0][1], int(item[0][2]))
    ):
        original_values = [float(row["original_wer"]) for row in group]
        updated_values = [float(row["updated_wer"]) for row in group]
        mean_original = statistics.fmean(original_values)
        mean_updated = statistics.fmean(updated_values)
        std_updated = statistics.stdev(updated_values) if len(updated_values) > 1 else 0.0
        aggregate.append(
            {
                "reward": reward,
                "dataset_tag": dataset_tag,
                "epochs": epochs,
                "n": str(len(group)),
                "original_wer_mean": fmt(mean_original),
                "updated_wer_mean": fmt(mean_updated),
                "updated_wer_std": fmt(std_updated),
                "absolute_delta_mean": fmt(mean_updated - mean_original),
                "relative_delta_pct_mean": f"{((mean_updated - mean_original) / mean_original) * 100.0:.2f}"
                if mean_original
                else "",
            }
        )
    return aggregate


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    csv_path: Path,
    checkpoint: str,
    command: str,
    branch: str,
    commit: str,
    log_path: str,
    screen_log_path: str,
) -> None:
    complete = sum(row["status"] == "complete" for row in rows)
    missing = [row for row in rows if row["status"] != "complete"]
    lines = [
        "# ROB-132 Audio SSL Test-Set Fixed-Reward Eval",
        "",
        "## Metadata",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder",
        "- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs",
        "- Datasets: `tedlium` and `earnings22`; both `test` split",
        "- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Main log: `{log_path}`",
        f"- Screen log: `{screen_log_path}`",
        f"- Queued command: `{command}`",
        "",
        f"Completed cells: `{complete}/{len(rows)}`.",
        "",
        "## Aggregate",
        "",
        "| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows(rows):
        lines.append(
            "| {reward} | {dataset_tag} | {epochs} | {n} | {original_wer_mean} | {updated_wer_mean} | "
            "{updated_wer_std} | {absolute_delta_mean} | {relative_delta_pct_mean} |".format(**row)
        )
    if missing:
        lines.extend(["", "## Missing Cells", ""])
        for row in missing:
            lines.append("- reward {reward} / {dataset_tag} / epoch {epochs} / lr `{lr}`".format(**row))
    lines.extend(
        [
            "",
            "## Per Cell",
            "",
            "| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {reward} | {dataset_tag} | {epochs} | `{lr}` | {original_wer} | "
            "{updated_wer} | {wer_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )
    lines.extend(["", "CSV artifact:", "", f"```text\n{csv_path}\n```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/"
            "rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22"
        ),
    )
    parser.add_argument("--fixed-rewards", default="1.0 0.0")
    parser.add_argument("--datasets", default="tedlium earnings22")
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--screen-log-path", required=True)
    parser.add_argument("--csv-name", default="rob132_audio_ssl_self_train_test_sets_fixed_rewards.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    csv_path = args.result_root / args.csv_name
    rows = collect_rows(
        args.result_root,
        rewards=parse_strings(args.fixed_rewards),
        datasets=parse_strings(args.datasets),
        epochs=parse_ints(args.epochs),
        lr=args.lr,
    )
    write_csv(rows, csv_path)
    write_markdown(
        rows,
        args.result_root / args.outcome_name,
        csv_path=csv_path,
        checkpoint=args.checkpoint,
        command=args.command,
        branch=args.branch,
        commit=args.commit,
        log_path=args.log_path,
        screen_log_path=args.screen_log_path,
    )
    print(f"[rob132-testsets-summary] wrote {args.result_root / args.csv_name}")
    print(f"[rob132-testsets-summary] wrote {args.result_root / args.outcome_name}")
    print(f"[rob132-testsets-summary] completed {sum(row['status'] == 'complete' for row in rows)}/{len(rows)} cells")


if __name__ == "__main__":
    main()
