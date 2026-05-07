#!/usr/bin/env python3
"""Plot frequency-bin mask distributions for UFMR or random frequency masking."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CHECKPOINT = Path("/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/test_wer/model.pt")
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "UFMR/UFMR_mask.pdf"
DEFAULT_CSV = SCRIPT_DIR.parent / "UFMR/UFMR_mask.csv"

sys.path.insert(0, str(REPO_ROOT))

from l2augment.modelling.models import FrequencyMaskingRanker, UnconditionalFrequencyMaskingRanker  # noqa: E402


def load_ranker(checkpoint_path: Path) -> UnconditionalFrequencyMaskingRanker:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {})

    loss_type = "mse"
    try:
        loss_type = config.policy.loss_type
    except Exception:
        pass

    model = UnconditionalFrequencyMaskingRanker(loss_type=loss_type)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def estimate_mask_distribution(
    model: UnconditionalFrequencyMaskingRanker,
    repeats: int,
    num_samples: int,
    batch_size: int,
    mel_bins: int,
) -> torch.Tensor:
    masked_counts = torch.zeros(mel_bins)
    total = 0

    while total < num_samples:
        current_batch_size = min(batch_size, num_samples - total)
        dummy_audio = torch.ones(current_batch_size, mel_bins, 1)
        _, selected_masks = model.learnt_augmentation(dummy_audio, repeats=repeats)
        masked_counts += (1.0 - selected_masks).sum(dim=0).cpu()
        total += current_batch_size

    return masked_counts / float(num_samples)


@torch.no_grad()
def estimate_random_mask_distribution(
    repeats: int,
    num_samples: int,
    batch_size: int,
    mel_bins: int,
) -> torch.Tensor:
    del repeats
    policy = FrequencyMaskingRanker()
    masked_counts = torch.zeros(mel_bins)
    total = 0

    while total < num_samples:
        current_batch_size = min(batch_size, num_samples - total)
        dummy_audio = torch.ones(current_batch_size, mel_bins, 1)
        _, selected_masks = policy.augment(dummy_audio, use_random=True)
        masked_counts += (1.0 - selected_masks).sum(dim=0).cpu()
        total += current_batch_size

    return masked_counts / float(num_samples)


def write_csv(mask_probabilities: torch.Tensor, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_bin", "mask_probability"])
        for idx, value in enumerate(mask_probabilities.tolist()):
            writer.writerow([idx, f"{value:.10f}"])


def plot(mask_probabilities: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = list(range(mask_probabilities.numel()))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(bins, mask_probabilities.numpy(), width=0.9, color="#3b6ea8", edgecolor="black", linewidth=0.25)
    ax.set_xlabel("Mel-frequency bin")
    ax.set_ylabel("Selected mask probability")
    ax.set_xlim(-0.5, mask_probabilities.numel() - 0.5)
    ax.set_ylim(0.0, max(0.05, float(mask_probabilities.max()) * 1.12))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ufmr", "random"], default="ufmr")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--repeats", type=int, default=15, help="Candidate masks sampled per UFMR selection.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of selected masks to average.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mel-bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.mode == "ufmr":
        model = load_ranker(args.checkpoint)
        mask_probabilities = estimate_mask_distribution(
            model=model,
            repeats=args.repeats,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            mel_bins=args.mel_bins,
        )
    else:
        mask_probabilities = estimate_random_mask_distribution(
            repeats=args.repeats,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            mel_bins=args.mel_bins,
        )
    write_csv(mask_probabilities, args.csv)
    plot(mask_probabilities, args.output)

    print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
