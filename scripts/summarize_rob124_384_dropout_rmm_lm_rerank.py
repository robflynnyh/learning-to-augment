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


def load_rows(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_by_condition(rows, condition):
    for row in rows:
        if row.get("condition") == condition:
            return row
    return {}


def best_completed_row(rows):
    complete = [row for row in rows if row.get("status") == "complete" and row.get("updated_wer")]
    if not complete:
        return {}
    return min(complete, key=lambda row: float(row["updated_wer"]))


def float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--previous-rob124-csv", required=True)
    parser.add_argument("--rob120-csv", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--screen-log-path", required=True)
    parser.add_argument("--csv-name", default="rob124_384_dropout_rmm_lm_rerank.csv")
    parser.add_argument("--outcome-name", default="OUTCOME.md")
    args = parser.parse_args()

    root = Path(args.result_root)
    method = "RMMReward1LMRerank15"
    label = "RMM 15 candidates, reward-1 LM CE rerank"
    result_path = root / method / "earnings22_test_epoch1_lr1e-5.txt"
    config_path = root / method / "configs" / "earnings22_test_epoch1_lr1e-5.yaml"
    parsed = parse_result(result_path)
    status = "complete" if parsed else "missing"

    previous_rob124_rows = load_rows(args.previous_rob124_csv)
    rob120_rows = load_rows(args.rob120_csv)
    rob124_fixed1 = row_by_condition(previous_rob124_rows, "fixed_1.0")
    rob124_best = best_completed_row(previous_rob124_rows)
    rob120_fixed1 = row_by_condition(rob120_rows, "fixed_1.0")

    rob124_fixed1_updated = float_or_none(rob124_fixed1.get("updated_wer"))
    rob124_best_updated = float_or_none(rob124_best.get("updated_wer"))
    rob120_fixed1_updated = float_or_none(rob120_fixed1.get("updated_wer"))
    updated = parsed["updated_wer"] if parsed else None

    row = {
        "condition": "rmm_reward1_lm_rerank15",
        "method": method,
        "label": label,
        "candidate_repeats": "15",
        "scorer_reward": "1.0",
        "config_path": str(config_path),
        "result_path": str(result_path),
        "status": status,
        "original_wer": fmt(parsed["original_wer"]) if parsed else "",
        "updated_wer": fmt(updated),
        "wer_delta": fmt(parsed["wer_delta"]) if parsed else "",
        "relative_wer_change_pct": f"{parsed['relative_wer_change_pct']:.2f}" if parsed else "",
        "rob124_fixed1_updated_wer": fmt(rob124_fixed1_updated),
        "delta_vs_rob124_fixed1": fmt(updated - rob124_fixed1_updated) if updated is not None and rob124_fixed1_updated is not None else "",
        "rob124_best_condition": rob124_best.get("condition", ""),
        "rob124_best_updated_wer": fmt(rob124_best_updated),
        "delta_vs_rob124_best": fmt(updated - rob124_best_updated) if updated is not None and rob124_best_updated is not None else "",
        "rob120_fixed1_updated_wer": fmt(rob120_fixed1_updated),
        "delta_vs_rob120_fixed1": fmt(updated - rob120_fixed1_updated) if updated is not None and rob120_fixed1_updated is not None else "",
    }

    csv_path = root / args.csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    outcome_path = root / args.outcome_name
    lines = [
        "# ROB-124 384-Dropout RMM-LM Rerank Evaluation",
        "",
        "## Run Metadata",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        "- Policy: `RMMRewardConditionedMaskLMReranker`.",
        "- Rerank recipe: generate `15` RMM candidate masks per adaptation step, encode each mask with the mask BVAE, score each VQ sequence with the 384/dropout reward-conditioned mask LM at fixed reward `1.0`, and adapt with the lowest per-candidate CE-loss mask.",
        "- Dataset/split: `earnings22` / `test`",
        "- Adaptation: `epochs=1`, `lr=1e-5`, multistep rollout",
        f"- Previous ROB-124 comparison CSV: `{args.previous_rob124_csv}`",
        f"- ROB-120 comparison CSV: `{args.rob120_csv}`",
        f"- Branch: `{args.branch}`",
        f"- Commit: `{args.commit}`",
        f"- Main log: `{args.log_path}`",
        f"- Screen log: `{args.screen_log_path}`",
        f"- Queued command: `{args.command}`",
        "",
        "## Results",
        "",
        "| Method | Status | Original WER | Updated WER | Delta | vs ROB-124 fixed 1.0 | vs ROB-124 best prior | vs ROB-120 fixed 1.0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {label} | {status} | {original} | {updated} | {delta} | {d124_fixed} | {d124_best} | {d120_fixed} |".format(
            label=label,
            status=status,
            original=row["original_wer"],
            updated=row["updated_wer"],
            delta=row["wer_delta"],
            d124_fixed=row["delta_vs_rob124_fixed1"],
            d124_best=row["delta_vs_rob124_best"],
            d120_fixed=row["delta_vs_rob120_fixed1"],
        ),
        "",
        f"CSV artifact: `{csv_path}`",
    ]
    if status != "complete":
        lines.extend(
            [
                "",
                "Residual risk: this is a queued or partial snapshot; inspect the log before treating it as a final comparison.",
            ]
        )
    outcome_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[rob124-rerank-summary] wrote {csv_path}")
    print(f"[rob124-rerank-summary] wrote {outcome_path}")
    print(f"[rob124-rerank-summary] status={status}")


if __name__ == "__main__":
    main()
