#!/usr/bin/env python3
"""Visualize RAC-MLM average masks for several audio segments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/exp/exp4/acp21rjf/.scratch/matplotlib",
)

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent
    for parent in SCRIPT_PATH.parents
    if (parent / "l2augment" / "modelling" / "models.py").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from l2augment.utils.datasets import AudioRewardConditionedMaskLMDataset
from l2augment.utils.helpers import load_model as load_policy
from l2augment.utils.helpers import load_rl_models
from lcasr.utils.audio_tools import total_frames


DEFAULT_CONFIG = Path(
    "exp/configs/reward_conditioned_lm/audio_ssl_conditioning/"
    "tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/store/store5/data/acp21rjf_checkpoints/l2augment/models/"
    "reward_conditioned_mask_lm/"
    "audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt"
)
DEFAULT_ROLLOUT_ROOT = Path("/store/store4/data/l2augment_rollout_uvqmlm/train")
DEFAULT_TEDLIUM_BASE = Path("/store/store4/data/TEDLIUM_release-3/legacy")
DEFAULT_OUTPUT_DIR = Path(
    "exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/"
    "rob132_hubert_base_transformer384/visualizations/"
    "rac_mlm_average_masks_tedlium_test_5x1000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sampled RAC-MLM masks for several TEDLIUM audio segments, "
            "streaming only the average mask for each segment."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", choices=("tedlium-test", "rollout"), default="tedlium-test")
    parser.add_argument("--tedlium-base", type=Path, default=DEFAULT_TEDLIUM_BASE)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--rollout", action="append", type=Path, default=None)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--samples-per-rollout", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--conditioning-reward", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available by default.",
    )
    return parser.parse_args()


def select_distinct_recording_rollouts(root: Path, count: int) -> list[Path]:
    selected: list[Path] = []
    seen_recordings: set[str] = set()
    for path in sorted(root.glob("*.pt")):
        recording_id = path.stem.rsplit("_", 1)[0]
        if recording_id in seen_recordings:
            continue
        selected.append(path)
        seen_recordings.add(recording_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"Found only {len(selected)} distinct recordings under {root}, need {count}")


def rollout_recording_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def rollout_audio_length(path: Path) -> int | None:
    try:
        rollout = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    audio = rollout.get("audio")
    if audio is None:
        return None
    return int(audio.shape[-1])


def load_ssl_features_for_segment(
    helper: AudioRewardConditionedMaskLMDataset,
    *,
    sph_path: Path,
    start_s: float,
    end_s: float,
) -> torch.Tensor:
    bundle, ssl_model = helper._load_ssl_model()
    waveform, sample_rate = helper._load_waveform_segment(str(sph_path), start_s, end_s)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    waveform = waveform.to(helper.ssl_device)
    lengths = torch.tensor([waveform.size(-1)], dtype=torch.long, device=waveform.device)
    with torch.no_grad():
        extracted = ssl_model.extract_features(waveform, lengths=lengths)
    if isinstance(extracted, tuple):
        features, feature_lengths = extracted
    else:
        features, feature_lengths = extracted, None
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if feature_lengths is not None:
        features = features[:, : int(feature_lengths[0].item())]
    return features.squeeze(0).cpu().contiguous().to(dtype=torch.float16)


def select_tedlium_test_segments(base: Path, count: int) -> list[dict]:
    split_root = base / "test"
    selected: list[dict] = []
    for sph_path in sorted((split_root / "sph").glob("*.sph")):
        recording_id = sph_path.stem
        stm_path = split_root / "stm" / f"{recording_id}.stm"
        utterances = AudioRewardConditionedMaskLMDataset._parse_stm(stm_path)
        if not utterances:
            continue
        utterance = utterances[0]
        selected.append(
            {
                "source_kind": "tedlium",
                "split": "test",
                "recording_id": recording_id,
                "utterance_index": 0,
                "sph_path": sph_path,
                "stm_path": stm_path,
                "start": float(utterance["start"]),
                "end": float(utterance["end"]),
                "text": utterance["text"],
            }
        )
        if len(selected) == count:
            return selected
    raise ValueError(f"Found only {len(selected)} TED-LIUM test segments under {split_root}, need {count}")


def load_audio_item(config, rollout_path: Path, device: torch.device) -> dict:
    dataset_config = OmegaConf.to_container(config.get("dataset", {}), resolve=True)
    dataset_config["ssl_device"] = device.type
    dataset = AudioRewardConditionedMaskLMDataset([str(rollout_path)], **dataset_config)
    item = dataset[0]
    if item is None:
        raise RuntimeError(f"Failed to load audio-conditioned item for {rollout_path}")
    return item


def build_sample_specs(args: argparse.Namespace, config, model, device: torch.device) -> list[dict]:
    if args.source == "rollout":
        rollouts = args.rollout or select_distinct_recording_rollouts(args.rollout_root, args.num_rollouts)
        if len(rollouts) != args.num_rollouts:
            raise ValueError(f"Expected {args.num_rollouts} rollouts, got {len(rollouts)}")
        specs = []
        for rollout_path in rollouts:
            item = load_audio_item(config, rollout_path, device)
            specs.append(
                {
                    "source_kind": "rollout",
                    "split": rollout_path.parent.name,
                    "source_path": rollout_path,
                    "rollout": str(rollout_path),
                    "recording_id": rollout_recording_id(rollout_path),
                    "audio_features": item["audio_features"],
                    "audio_feature_length": item["audio_feature_length"],
                    "target_prediction_steps": int(item["generation_length"]),
                    "target_output_length": rollout_audio_length(rollout_path),
                }
            )
        return specs

    dataset_config = OmegaConf.to_container(config.get("dataset", {}), resolve=True)
    dataset_config["ssl_device"] = device.type
    helper = AudioRewardConditionedMaskLMDataset([], **dataset_config)
    specs = []
    for segment in select_tedlium_test_segments(args.tedlium_base, args.num_rollouts):
        audio_features = load_ssl_features_for_segment(
            helper,
            sph_path=segment["sph_path"],
            start_s=segment["start"],
            end_s=segment["end"],
        )
        duration_s = segment["end"] - segment["start"]
        target_output_length = total_frames(duration_s)
        target_prediction_steps = int(model.mask_enc.calc_downsampled_length(target_output_length).item())
        specs.append(
            {
                **segment,
                "source_path": segment["sph_path"],
                "audio_features": audio_features,
                "audio_feature_length": torch.tensor(audio_features.shape[0], dtype=torch.long),
                "target_prediction_steps": target_prediction_steps,
                "target_output_length": target_output_length,
                "duration_seconds": duration_s,
            }
        )
    return specs


@torch.no_grad()
def generate_mask_batch(
    model,
    *,
    audio_features: torch.Tensor,
    audio_feature_length: torch.Tensor,
    conditioning_reward: float,
    batch_size: int,
    target_prediction_steps: int,
    target_output_length: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if audio_features.dim() == 2:
        audio_features = audio_features.unsqueeze(0)
    audio_features = audio_features.to(device)
    audio_feature_lengths = audio_feature_length.reshape(1).to(device)
    audio_item_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)
    reward = torch.full((batch_size,), float(conditioning_reward), dtype=torch.float32, device=device)

    outputs: list[torch.Tensor] = []
    for step in range(target_prediction_steps):
        if outputs:
            previous = torch.stack(outputs, dim=1)
            generations = torch.cat(
                (previous, torch.zeros(batch_size, 1, dtype=torch.long, device=device)),
                dim=1,
            )
        else:
            generations = torch.empty(batch_size, 1, dtype=torch.long, device=device)
        generation_lengths = torch.full((batch_size,), step + 1, dtype=torch.long, device=device)
        pred, _, _ = model(
            generations=generations,
            generation_lengths=generation_lengths,
            rewards=reward,
            audio_features=audio_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_item_idxs=audio_item_idxs,
        )
        pred = pred[:, step : step + 1]
        pred[..., model.codebook_size] = -torch.finfo(pred.dtype).max
        probs = pred.softmax(-1).squeeze(1)
        outputs.append(torch.multinomial(probs, 1).squeeze(-1))

    tokens = torch.stack(outputs, dim=1)
    mask_latent = model.mask_enc.VQ.codebook[tokens].transpose(-1, -2)
    mask_h = model.mask_enc.latent_to_hidden(mask_latent)
    mask_h = model.mask_enc.rnn_out(mask_h) + mask_h
    mask_h = model.mask_enc.decoder(mask_h)
    if target_output_length is not None:
        mask_h = torch.nn.functional.interpolate(
            mask_h,
            size=target_output_length,
            mode="linear",
            align_corners=False,
        )
    masks = model.mask_enc.output(mask_h).sigmoid()
    masks = torch.round(masks, decimals=0).to(dtype=torch.float32)
    return masks, tokens


def save_grid(summaries: list[dict], pdf_path: Path, png_path: Path) -> None:
    fig, axes = plt.subplots(len(summaries), 1, figsize=(10.0, 2.35 * len(summaries)), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    image = None
    for index, (ax, summary) in enumerate(zip(axes, summaries, strict=True), start=1):
        masked_rate = 1.0 - summary["average_keep_mask"]
        image = ax.imshow(
            masked_rate,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(
            f"Sample {index}: {summary['recording_id']} "
            f"(masked {summary['average_masked_percentage']:.1f}%)"
        )
        ax.set_xlabel("time frame")
        ax.set_ylabel("mel bin")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.tolist(), label="masked probability")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.samples_per_rollout < 1:
        raise ValueError("--samples-per-rollout must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    config = OmegaConf.load(args.config)
    model = load_rl_models(config).to(device)
    load_policy(model, config, path=str(args.checkpoint))
    model.eval()
    model.mask_enc.eval()

    sample_specs = build_sample_specs(args, config, model, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    npz_payload: dict[str, np.ndarray] = {}

    for sample_index, sample_spec in enumerate(sample_specs, start=1):
        audio_features = sample_spec["audio_features"].to(device)
        audio_feature_length = sample_spec["audio_feature_length"].to(device)
        target_prediction_steps = int(sample_spec["target_prediction_steps"])
        target_output_length = sample_spec["target_output_length"]

        torch.manual_seed(args.seed + sample_index * 100_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + sample_index * 100_000)

        generated = 0
        running_average: torch.Tensor | None = None
        sample_keep_sum = 0.0
        sample_keep_min = float("inf")
        sample_keep_max = float("-inf")
        example_tokens: list[list[int]] = []

        while generated < args.samples_per_rollout:
            current_batch = min(args.batch_size, args.samples_per_rollout - generated)
            masks, tokens = generate_mask_batch(
                model,
                audio_features=audio_features,
                audio_feature_length=audio_feature_length,
                conditioning_reward=args.conditioning_reward,
                batch_size=current_batch,
                target_prediction_steps=target_prediction_steps,
                target_output_length=target_output_length,
                device=device,
            )
            masks_cpu = masks.detach().cpu()
            if running_average is None:
                running_average = torch.zeros(masks_cpu.shape[1:], dtype=torch.float64)
            running_average += masks_cpu.sum(dim=0).to(dtype=torch.float64) / args.samples_per_rollout

            keep_fraction = masks_cpu.mean(dim=(1, 2))
            sample_keep_sum += float(keep_fraction.sum().item())
            sample_keep_min = min(sample_keep_min, float(keep_fraction.min().item()))
            sample_keep_max = max(sample_keep_max, float(keep_fraction.max().item()))
            for row in tokens.detach().cpu()[: max(0, 3 - len(example_tokens))]:
                example_tokens.append([int(item) for item in row.tolist()])
            generated += current_batch
            print(
                f"sample={sample_index}/{len(sample_specs)} generated={generated}/{args.samples_per_rollout}",
                flush=True,
            )

        if running_average is None:
            raise RuntimeError(f"No masks generated for {sample_spec['source_path']}")

        average_keep_mask = running_average.numpy().astype(np.float32)
        sample_key = f"sample_{sample_index}"
        npz_payload[sample_key] = average_keep_mask
        average_keep_fraction = float(average_keep_mask.mean())
        sample_keep_mean = sample_keep_sum / args.samples_per_rollout
        summaries.append(
            {
                "sample_index": sample_index,
                "source_kind": sample_spec["source_kind"],
                "split": sample_spec["split"],
                "source_path": str(sample_spec["source_path"]),
                "recording_id": sample_spec["recording_id"],
                "conditioning_reward": float(args.conditioning_reward),
                "samples": int(args.samples_per_rollout),
                "target_prediction_steps": target_prediction_steps,
                "target_output_length": target_output_length,
                "audio_feature_shape": list(sample_spec["audio_features"].shape),
                "audio_feature_length": int(sample_spec["audio_feature_length"]),
                "average_keep_mask": average_keep_mask,
                "average_keep_fraction": average_keep_fraction,
                "average_keep_percentage": average_keep_fraction * 100.0,
                "average_masked_fraction": 1.0 - average_keep_fraction,
                "average_masked_percentage": (1.0 - average_keep_fraction) * 100.0,
                "sample_keep_fraction_min": sample_keep_min,
                "sample_keep_fraction_mean": sample_keep_mean,
                "sample_keep_fraction_max": sample_keep_max,
                "sample_masked_fraction_mean": 1.0 - sample_keep_mean,
                "example_token_sequences": example_tokens,
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in sample_spec.items()
                    if key
                    in {
                        "rollout",
                        "utterance_index",
                        "sph_path",
                        "stm_path",
                        "start",
                        "end",
                        "duration_seconds",
                        "text",
                    }
                },
            }
        )

    source_label = "tedlium_test" if args.source == "tedlium-test" else "rollout"
    artifact_stem = f"rac_mlm_average_masks_{source_label}_{len(sample_specs)}x{args.samples_per_rollout}"
    pdf_path = args.output_dir / f"{artifact_stem}.pdf"
    png_path = args.output_dir / f"{artifact_stem}.png"
    save_grid(summaries, pdf_path, png_path)
    npz_path = args.output_dir / f"{artifact_stem}.npz"
    np.savez_compressed(npz_path, **npz_payload)

    metadata = {
        "method": "RAC-MLM",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "source": args.source,
        "tedlium_base": str(args.tedlium_base),
        "rollout_root": str(args.rollout_root),
        "conditioning_reward": float(args.conditioning_reward),
        "samples_per_rollout": int(args.samples_per_rollout),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(device),
        "mask_value_semantics": (
            "Model output is a keep mask: 1.0 keeps/unmasks the bin. "
            "The PDF plots 1.0 - average_keep_mask as masked probability."
        ),
        "pdf": pdf_path.name,
        "png": png_path.name,
        "npz": npz_path.name,
        "samples": [
            {key: value for key, value in summary.items() if key != "average_keep_mask"}
            for summary in summaries
        ],
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
