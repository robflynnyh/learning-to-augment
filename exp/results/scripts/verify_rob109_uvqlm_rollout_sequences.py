#!/usr/bin/env python3
"""Verify ROB-109 saved UVQLM rollout code sequences against Mimas checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROLLOUT_ROOT = Path("/store/store4/data/l2augment_rollout_uvqmlm/dev")
DEFAULT_UMLM_CHECKPOINT = Path(
    "/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt"
)
DEFAULT_BVAE_CHECKPOINT = Path(
    "/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/"
    "bvae_USINGTHISFORNOW_2048gpu.pt"
)
DEFAULT_VECTOR_QUANTIZE_SITE = Path(
    "/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/"
    "lib/python3.9/site-packages"
)
DEFAULT_OUTPUT_JSON = Path(
    "exp/results/repro/unconditional_lm/ROB-109_rollout_verification/"
    "uvqlm_sequence_verification.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode saved UVQLM VQ sequences through the Mimas BVAE checkpoint "
            "and score them under the Mimas UMLM checkpoint."
        )
    )
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--umlm-checkpoint", type=Path, default=DEFAULT_UMLM_CHECKPOINT)
    parser.add_argument("--bvae-checkpoint", type=Path, default=DEFAULT_BVAE_CHECKPOINT)
    parser.add_argument(
        "--vector-quantize-site",
        type=Path,
        default=DEFAULT_VECTOR_QUANTIZE_SITE,
        help="Site-packages directory containing vector_quantize_pytorch.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Number of sorted rollout files to verify. Use 0 to verify all files.",
    )
    parser.add_argument("--seed", type=int, default=109)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def install_import_shims(vector_quantize_site: Path) -> None:
    if str(vector_quantize_site) not in sys.path:
        sys.path.append(str(vector_quantize_site))

    # The model module imports these lcasr utilities, but the UVQLM/BVAE path
    # used here does not execute them. Shimming avoids a torchaudio ABI mismatch
    # in the torch 2.8 environment required to read float8 rollout tensors.
    lcasr = types.ModuleType("lcasr")
    components = types.ModuleType("lcasr.components")
    batchrenorm = types.ModuleType("lcasr.components.batchrenorm")
    batchrenorm.BatchRenorm1d = nn.BatchNorm1d
    utils = types.ModuleType("lcasr.utils")
    augmentation = types.ModuleType("lcasr.utils.augmentation")

    class SpecAugment(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            return audio

    augmentation.SpecAugment = SpecAugment
    sys.modules.update(
        {
            "lcasr": lcasr,
            "lcasr.components": components,
            "lcasr.components.batchrenorm": batchrenorm,
            "lcasr.utils": utils,
            "lcasr.utils.augmentation": augmentation,
        }
    )

    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.log = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb


def patch_torch_load() -> None:
    original_load = torch.load

    def load_trusted_local_artifact(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_trusted_local_artifact


def load_model(args: argparse.Namespace):
    install_import_shims(args.vector_quantize_site)
    patch_torch_load()

    from l2augment.modelling.models import UnconditionalMaskGenerator

    model = UnconditionalMaskGenerator(
        mask_vae_state_dict_path=str(args.bvae_checkpoint),
        mask_vae_config={
            "latent_dim": 128,
            "codebook_size": 2048,
            "use_vq": True,
        },
    )
    state = torch.load(args.umlm_checkpoint, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def decode_codes_to_mask(model, codes: torch.Tensor, target_length: int) -> torch.Tensor:
    codes = codes.to(torch.long)
    mask_latent = model.mask_enc.VQ.codebook[codes][None].transpose(-1, -2)
    mask_h = model.mask_enc.latent_to_hidden(mask_latent)
    mask_h = model.mask_enc.rnn_out(mask_h) + mask_h
    mask_h = model.mask_enc.decoder(mask_h)
    mask_h = torch.nn.functional.interpolate(
        mask_h,
        size=int(target_length),
        mode="linear",
        align_corners=False,
    )
    mask_pred = model.mask_enc.output(mask_h).sigmoid()
    return torch.round(mask_pred, decimals=0).to(dtype=torch.long)


@torch.no_grad()
def sequence_nll(model, codes: torch.Tensor) -> float:
    codes = codes.to(torch.long)[None]
    lengths = torch.tensor([codes.shape[1]], dtype=torch.long)
    mask_emb = model.embeddings(codes)
    predictions, _, _ = model(mask_emb, lengths)
    eos = torch.full((1, 1), model.codebook_size, dtype=torch.long)
    target = torch.cat([codes, eos], dim=1)
    loss = F.cross_entropy(predictions.transpose(-1, -2), target, reduction="mean")
    return float(loss.detach().cpu())


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)),
    }


def main() -> None:
    args = parse_args()
    if not args.rollout_root.exists():
        raise FileNotFoundError(args.rollout_root)
    for checkpoint in (args.umlm_checkpoint, args.bvae_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    files = sorted(args.rollout_root.glob("*.pt"))
    total_available = len(files)
    if args.max_files < 0:
        raise ValueError("--max-files must be non-negative")
    if args.max_files:
        files = files[: args.max_files]

    model = load_model(args)
    random_generator = torch.Generator().manual_seed(args.seed)

    sequence_losses: list[float] = []
    random_losses: list[float] = []
    mismatch_rates: list[float] = []
    mismatch_pixels = 0
    total_pixels = 0
    examples = []
    nonzero_mismatch_examples = []

    for file_index, path in enumerate(files, start=1):
        data = torch.load(path, map_location="cpu")
        generations = data["generation"]
        masks = data["mask"]
        target_length = masks.shape[-1]

        for sequence_index, codes in enumerate(generations):
            reconstructed = decode_codes_to_mask(model, codes, target_length)
            saved = masks[sequence_index].to(torch.float32).round().to(torch.long)
            mismatch = reconstructed != saved
            mismatch_count = int(mismatch.sum().item())
            pixel_count = int(mismatch.numel())
            mismatch_pixels += mismatch_count
            total_pixels += pixel_count
            mismatch_rate = mismatch_count / pixel_count
            mismatch_rates.append(mismatch_rate)

            loss = sequence_nll(model, codes)
            random_codes = torch.randint(
                low=0,
                high=model.codebook_size,
                size=codes.shape,
                generator=random_generator,
            )
            random_loss = sequence_nll(model, random_codes)
            sequence_losses.append(loss)
            random_losses.append(random_loss)

            if len(examples) < 5:
                examples.append(
                    {
                        "file": path.name,
                        "sequence_index": sequence_index,
                        "code_count": int(codes.numel()),
                        "mismatch_pixels": mismatch_count,
                        "total_pixels": pixel_count,
                        "mask_mismatch_rate": mismatch_rate,
                        "sequence_nll": loss,
                        "random_code_nll": random_loss,
                    }
                )
            if mismatch_count and len(nonzero_mismatch_examples) < 5:
                nonzero_mismatch_examples.append(
                    {
                        "file": path.name,
                        "sequence_index": sequence_index,
                        "code_count": int(codes.numel()),
                        "mismatch_pixels": mismatch_count,
                        "total_pixels": pixel_count,
                        "mask_mismatch_rate": mismatch_rate,
                    }
                )

        if file_index == 1 or file_index % 25 == 0 or file_index == len(files):
            print(f"verified {file_index}/{len(files)} files", flush=True)

    summary = {
        "rollout_root": str(args.rollout_root),
        "total_available_files": total_available,
        "verified_files": len(files),
        "verified_sequences": len(sequence_losses),
        "umlm_checkpoint": str(args.umlm_checkpoint),
        "bvae_checkpoint": str(args.bvae_checkpoint),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "mask_reconstruction_mismatch_pixels": mismatch_pixels,
        "mask_reconstruction_total_pixels": total_pixels,
        "mask_reconstruction_mismatch_rate": summarize(mismatch_rates),
        "sequence_nll": summarize(sequence_losses),
        "random_code_nll": summarize(random_losses),
        "examples": examples,
        "nonzero_mismatch_examples": nonzero_mismatch_examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
