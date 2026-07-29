#!/usr/bin/env python3
"""Summarize ROB-339 RAC-MLM fixed-reward repeats."""

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
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
    "tal": ("this_american_life", "test"),
    "this_american_life": ("this_american_life", "test"),
    "chime6": ("chime6", "test"),
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


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split())


def parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(raw.split())


def reward_tag(reward: str) -> str:
    return reward.replace(".", "p").replace("-", "m")


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def row_paths(
    historical_root: Path,
    result_root: Path,
    method: str,
    dataset_tag: str,
    split: str,
    reward: str,
    epoch_count: int,
    lr: str,
    repeat: int,
) -> tuple[Path, Path]:
    reward_suffix = reward_tag(reward)
    if repeat == 1:
        tag = f"{dataset_tag}_{split}_reward{reward_suffix}_epoch{epoch_count}_lr{lr}"
        return (
            historical_root / method / "configs" / f"{tag}.yaml",
            historical_root / method / f"{tag}.txt",
        )

    tag = f"{dataset_tag}_{split}_reward{reward_suffix}_epoch{epoch_count}_lr{lr}_repeat{repeat}"
    return (
        result_root / method / "configs" / f"{tag}.yaml",
        result_root / method / f"{tag}.txt",
    )


def expected_rows(
    historical_root: Path,
    result_root: Path,
    rewards: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    lr: str,
    repeats: tuple[int, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for reward in rewards:
        method = f"AudioRewardConditionedMaskLMReward{reward_tag(reward)}"
        for dataset_tag in datasets:
            dataset, split = DATASETS[dataset_tag]
            for epoch_count in epochs:
                for repeat in repeats:
                    seed = 123456 + repeat - 1
                    config_path, result_path = row_paths(
                        historical_root=historical_root,
                        result_root=result_root,
                        method=method,
                        dataset_tag=dataset_tag,
                        split=split,
                        reward=reward,
                        epoch_count=epoch_count,
                        lr=lr,
                        repeat=repeat,
                    )
                    rows.append(
                        {
                            "condition": f"fixed_reward_{reward}",
                            "label": f"fixed conditioning reward {reward}",
                            "reward": reward,
                            "dataset_tag": dataset_tag,
                            "dataset": dataset,
                            "split": split,
                            "method": method,
                            "repeat": str(repeat),
                            "seed": str(seed),
                            "epochs": str(epoch_count),
                            "lr": lr,
                            "config_path": display_path(config_path),
                            "result_path": display_path(result_path),
                        }
                    )
    return rows


def collect_rows(
    historical_root: Path,
    result_root: Path,
    rewards: tuple[str, ...],
    datasets: tuple[str, ...],
    epochs: tuple[int, ...],
    lr: str,
    repeats: tuple[int, ...],
) -> list[dict[str, str]]:
    rows = expected_rows(historical_root, result_root, rewards, datasets, epochs, lr, repeats)
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
        if row["status"] == "complete":
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


def write_aggregate_csv(rows: list[dict[str, str]], path: Path) -> None:
    aggregate = aggregate_rows(rows)
    fieldnames = [
        "reward",
        "dataset_tag",
        "epochs",
        "n",
        "original_wer_mean",
        "updated_wer_mean",
        "updated_wer_std",
        "absolute_delta_mean",
        "relative_delta_pct_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate)


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    csv_path: Path,
    aggregate_csv_path: Path,
    historical_root: Path,
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
        "# ROB-339 RAC-MLM Fixed-Reward Repeats",
        "",
        "## Metadata",
        "",
        f"- Historical repeat 1 root: `{historical_root}`",
        f"- New repeat root: `{path.parent}`",
        f"- Checkpoint: `{checkpoint}`",
        "- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder",
        "- Repeat seeds: repeat 1 `123456`, repeat 2 `123457`, repeat 3 `123458`",
        "- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs",
        "- Datasets: `tedlium`, `earnings22`, `rev16`, `TAL`, `chime6`; all `test` split",
        "- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Main log: `{log_path}`",
        f"- Screen log: `{screen_log_path}`",
        f"- Queued command: `{command}`",
        "",
        f"Completed repeat rows: `{complete}/{len(rows)}`.",
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
            lines.append(
                "- reward {reward} / {dataset_tag} / epoch {epochs} / repeat {repeat} / seed {seed} / lr `{lr}`".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Per Repeat",
            "",
            "| Reward | Dataset | Epochs | Repeat | Seed | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {reward} | {dataset_tag} | {epochs} | {repeat} | {seed} | `{lr}` | {original_wer} | "
            "{updated_wer} | {wer_delta} | {relative_delta_pct} | {status} | `{result_path}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- Per-repeat CSV: `{csv_path}`",
            f"- Aggregate CSV: `{aggregate_csv_path}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=Path(
            "exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/"
            "rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/"
            "rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1_rob339_repeats"
        ),
    )
    parser.add_argument("--fixed-rewards", default="1.0 0.0")
    parser.add_argument("--datasets", default="tedlium earnings22 rev16 TAL chime6")
    parser.add_argument("--epochs", default="1 5")
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--repeats", default="1 2 3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--screen-log-path", required=True)
    parser.add_argument("--csv-name", default="rob339_rac_mlm_fixed_reward_repeats.csv")
    parser.add_argument("--aggregate-csv-name", default="rob339_rac_mlm_fixed_reward_repeats_aggregate.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    rows = collect_rows(
        historical_root=args.historical_root,
        result_root=args.result_root,
        rewards=parse_strings(args.fixed_rewards),
        datasets=parse_strings(args.datasets),
        epochs=parse_ints(args.epochs),
        lr=args.lr,
        repeats=parse_ints(args.repeats),
    )
    csv_path = args.result_root / args.csv_name
    aggregate_csv_path = args.result_root / args.aggregate_csv_name
    write_csv(rows, csv_path)
    write_aggregate_csv(rows, aggregate_csv_path)
    write_markdown(
        rows,
        args.result_root / args.outcome_name,
        csv_path=csv_path,
        aggregate_csv_path=aggregate_csv_path,
        historical_root=args.historical_root,
        checkpoint=args.checkpoint,
        command=args.command,
        branch=args.branch,
        commit=args.commit,
        log_path=args.log_path,
        screen_log_path=args.screen_log_path,
    )
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[rob339-summary] wrote {csv_path}")
    print(f"[rob339-summary] wrote {aggregate_csv_path}")
    print(f"[rob339-summary] wrote {args.result_root / args.outcome_name}")
    print(f"[rob339-summary] completed {complete}/{len(rows)} repeat rows")


if __name__ == "__main__":
    main()
