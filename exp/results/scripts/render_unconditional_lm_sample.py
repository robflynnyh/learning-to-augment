#!/usr/bin/env python3
"""Render unconditional VQ mask LM samples as viewable artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/exp/exp4/acp21rjf/.scratch/matplotlib-cache",
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import vector_quantize_pytorch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

_vq_init_signature = inspect.signature(vector_quantize_pytorch.VectorQuantize.__init__)
if "rotation_trick" not in _vq_init_signature.parameters:
    _original_vq_init = vector_quantize_pytorch.VectorQuantize.__init__

    def _vq_init_compat(self, *args, **kwargs):
        kwargs.pop("rotation_trick", None)
        return _original_vq_init(self, *args, **kwargs)

    vector_quantize_pytorch.VectorQuantize.__init__ = _vq_init_compat

from l2augment.modelling.models import UnconditionalMaskGenerator


DEFAULT_UMLM_CHECKPOINT = Path(
    "/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt"
)
DEFAULT_BVAE_CHECKPOINT = Path(
    "/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/"
    "bvae_USINGTHISFORNOW_2048gpu.pt"
)
DEFAULT_OUTPUT_DIR = Path("exp/results/repro/unconditional_lm/ROB-73_sample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the unconditional VQ mask LM and render its binary mask."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--umlm-checkpoint", type=Path, default=DEFAULT_UMLM_CHECKPOINT)
    parser.add_argument("--bvae-checkpoint", type=Path, default=DEFAULT_BVAE_CHECKPOINT)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of unconditional mask samples to render.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax decoding instead of multinomial sampling.",
    )
    return parser.parse_args()


def load_model(umlm_checkpoint: Path, bvae_checkpoint: Path) -> UnconditionalMaskGenerator:
    for checkpoint in (umlm_checkpoint, bvae_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = UnconditionalMaskGenerator(
        mask_vae_state_dict_path=str(bvae_checkpoint),
        mask_vae_config={
            "latent_dim": 128,
            "codebook_size": 2048,
            "use_vq": True,
        },
    )
    state = torch.load(umlm_checkpoint, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def save_mask_figure(mask: np.ndarray, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.imshow(
        mask,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel bin")
    ax.set_title("Unconditional VQ mask LM sample")
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_mask_grid(masks: list[np.ndarray], path_base: Path) -> None:
    cols = min(5, len(masks))
    rows = int(np.ceil(len(masks) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.0 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx >= len(masks):
            ax.axis("off")
            continue
        ax.imshow(
            masks[idx],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"sample {idx:02d}")
        ax.set_xlabel("time")
        ax.set_ylabel("mel")
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_masked_toy_spectrogram(mask: np.ndarray, path_base: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = mask.shape[1]
    time = np.linspace(0, 1, frames, dtype=np.float32)[None, :]
    mel = np.linspace(0, 1, mask.shape[0], dtype=np.float32)[:, None]
    toy_spec = 0.45 + 0.35 * np.sin(2 * np.pi * (4.0 * time + 1.5 * mel))
    toy_spec += 0.12 * rng.standard_normal(mask.shape)
    masked = np.clip(toy_spec * mask, 0, 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), constrained_layout=True)
    axes[0].imshow(toy_spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Toy spectrogram before mask")
    axes[0].set_ylabel("mel bin")
    axes[1].imshow(masked, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Toy spectrogram after generated mask")
    axes[1].set_xlabel("time frame")
    axes[1].set_ylabel("mel bin")
    for suffix in (".png", ".pdf"):
        fig.savefig(path_base.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    torch.manual_seed(args.seed)
    model = load_model(args.umlm_checkpoint, args.bvae_checkpoint)
    lengths = torch.tensor([args.frames], dtype=torch.long)
    prediction_steps = int(model.mask_enc.calc_downsampled_length(lengths)[0].item())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    sample_metadata = []
    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            mask_pred, generation = model.generate(
                sample=not args.deterministic,
                target_output_length=args.frames,
                target_prediction_steps=prediction_steps,
                device="cpu",
            )

            mask = mask_pred.squeeze(0).cpu().numpy().astype(np.float32)
            codes = generation.cpu().tolist()
            masks.append(mask)

            sample_stem = f"sample_{sample_idx:02d}"
            save_mask_figure(mask, samples_dir / f"{sample_stem}_mask")
            save_masked_toy_spectrogram(
                mask,
                samples_dir / f"{sample_stem}_toy_spectrogram_masked",
                seed=args.seed + sample_idx,
            )
            sample_metadata.append(
                {
                    "sample_index": sample_idx,
                    "mask_path_png": f"samples/{sample_stem}_mask.png",
                    "mask_path_pdf": f"samples/{sample_stem}_mask.pdf",
                    "toy_spectrogram_path_png": (
                        f"samples/{sample_stem}_toy_spectrogram_masked.png"
                    ),
                    "toy_spectrogram_path_pdf": (
                        f"samples/{sample_stem}_toy_spectrogram_masked.pdf"
                    ),
                    "generated_code_count": len(codes),
                    "mask_keep_fraction": float(mask.mean()),
                    "generated_vq_codes": codes,
                }
            )

    save_mask_grid(masks, args.output_dir / "unconditional_vq_mask_samples_grid")

    metadata = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "frames": args.frames,
        "mel_bins": int(masks[0].shape[0]),
        "prediction_steps": prediction_steps,
        "decoding": "argmax" if args.deterministic else "multinomial_sample",
        "umlm_checkpoint": str(args.umlm_checkpoint),
        "bvae_checkpoint": str(args.bvae_checkpoint),
        "samples": sample_metadata,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# ROB-73 Unconditional VQ Mask LM Samples",
                "",
                "Generated by `exp/results/scripts/render_unconditional_lm_sample.py`.",
                "",
                "Artifacts:",
                "",
                "- `unconditional_vq_mask_samples_grid.png` / `.pdf`: overview of all",
                "  sampled binary masks decoded by the trained unconditional VQ mask LM.",
                "- `samples/sample_XX_mask.png` / `.pdf`: each individual sampled binary",
                "  mask.",
                "- `samples/sample_XX_toy_spectrogram_masked.png` / `.pdf`: the same mask",
                "  applied to a synthetic spectrogram only to make the kept/dropped regions",
                "  easier to view.",
                "- `metadata.json`: seed, frame count, checkpoint paths, keep fraction, and",
                "  sampled VQ code sequence for each sample.",
                "",
                "The model output itself is the binary 80-by-time mask. The toy spectrogram",
                "is not model-generated audio; it is a visualization aid.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote artifacts to {args.output_dir}")
    keep_fractions = [sample["mask_keep_fraction"] for sample in sample_metadata]
    print(
        f"num_samples={args.num_samples} mask_shape={masks[0].shape} "
        f"keep_fraction_range={min(keep_fractions):.4f}-{max(keep_fractions):.4f}"
    )
    print(f"prediction_steps={prediction_steps}")


if __name__ == "__main__":
    main()
