#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(
    r"ID: (?P<id>.*?) - Dataset: (?P<dataset>.*?) - Split: (?P<split>.*?) "
    r"- Epochs: (?P<epochs>\d+) - Original_WER: (?P<original>[0-9.eE+-]+) "
    r"- Updated_WER: (?P<updated>[0-9.eE+-]+)"
)


CONDITIONS = (
    ("fixed_0.0", "RewardConditionedMaskLMReward0", "fixed reward 0.0"),
    ("fixed_1.0", "RewardConditionedMaskLMReward1", "fixed reward 1.0"),
    ("uniform_0.0_1.0", "RewardConditionedMaskLMUniform0to1", "uniform reward [0.0, 1.0]"),
    ("uniform_0.5_1.0", "RewardConditionedMaskLMUniform0p5to1", "uniform reward [0.5, 1.0]"),
)


def parse_result(path):
    if not path.exists():
        return None
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RESULT_RE.search(line)
        if match:
            matches.append(match)
    if not matches:
        return None
    match = matches[-1]
    original = float(match.group("original"))
    updated = float(match.group("updated"))
    return {
        "id": match.group("id"),
        "dataset": match.group("dataset"),
        "split": match.group("split"),
        "epochs": int(match.group("epochs")),
        "original_wer": original,
        "updated_wer": updated,
        "wer_delta": updated - original,
        "relative_wer_change_pct": ((updated - original) / original * 100.0) if original else 0.0,
    }


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--screen-log-path", required=True)
    parser.add_argument("--csv-name", default="rob120_earnings_reward_controls.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = []
    for condition_key, method, label in CONDITIONS:
        result_path = root / method / "earnings22_test_epoch1_lr1e-5.txt"
        config_path = root / method / "configs" / "earnings22_test_epoch1_lr1e-5.yaml"
        parsed = parse_result(result_path)
        status = "complete" if parsed else "missing"
        row = {
            "condition": condition_key,
            "method": method,
            "label": label,
            "config_path": str(config_path),
            "result_path": str(result_path),
            "status": status,
            "original_wer": "",
            "updated_wer": "",
            "wer_delta": "",
            "relative_wer_change_pct": "",
        }
        if parsed:
            row.update(
                {
                    "original_wer": fmt(parsed["original_wer"]),
                    "updated_wer": fmt(parsed["updated_wer"]),
                    "wer_delta": fmt(parsed["wer_delta"]),
                    "relative_wer_change_pct": f"{parsed['relative_wer_change_pct']:.2f}",
                }
            )
        rows.append(row)

    csv_path = root / args.csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    complete_count = sum(row["status"] == "complete" for row in rows)
    outcome_path = root / args.outcome_name
    lines = [
        "# ROB-120 Earnings Reward-Control Evaluation",
        "",
        "## Run Metadata",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        "- Dataset/split: `earnings22` / `test`",
        "- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout",
        f"- Branch: `{args.branch}`",
        f"- Commit: `{args.commit}`",
        f"- Main log: `{args.log_path}`",
        f"- Screen log: `{args.screen_log_path}`",
        f"- Queued command: `{args.command}`",
        "",
        "## Results",
        "",
        "| Condition | Status | Config | Result | Original WER | Updated WER | Delta | Relative change |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rel = f"{row['relative_wer_change_pct']}%" if row["relative_wer_change_pct"] else ""
        lines.append(
            "| {label} | {status} | `{config}` | `{result}` | {original} | {updated} | {delta} | {rel} |".format(
                label=row["label"],
                status=row["status"],
                config=row["config_path"],
                result=row["result_path"],
                original=row["original_wer"],
                updated=row["updated_wer"],
                delta=row["wer_delta"],
                rel=rel,
            )
        )
    lines.extend(
        [
            "",
            f"Completed conditions: `{complete_count}/4`.",
            "",
            "CSV artifact:",
            "",
            f"```text\n{csv_path}\n```",
        ]
    )
    if complete_count < 4:
        lines.extend(
            [
                "",
                "Residual risk: this is a partial snapshot; inspect the log before treating it as the final comparison.",
            ]
        )
    outcome_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[rob120-summary] wrote {csv_path}")
    print(f"[rob120-summary] wrote {outcome_path}")
    print(f"[rob120-summary] completed {complete_count}/4 conditions")


if __name__ == "__main__":
    main()
