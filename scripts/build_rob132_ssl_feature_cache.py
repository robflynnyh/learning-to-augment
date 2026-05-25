#!/usr/bin/env python3
"""Build mask-token-aligned SSL feature sidecars for ROB-132 rollouts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F


def parse_stm(path: Path) -> list[dict[str, object]]:
    utterances = []
    with path.open("r") as handle:
        for line in handle:
            parts = line.strip().split(" ")
            if len(parts) < 7:
                continue
            _, _, _, start, end, _, *text = parts
            text = " ".join(text)
            if text == "ignore_time_segment_in_scoring":
                continue
            text = re.sub(r"<[^>]*>", "", text)
            utterances.append({"start": float(start), "end": float(end), "text": text})
    return utterances


def rollout_to_tedlium_paths(rollout_path: Path, tedlium_base: Path) -> tuple[Path, Path, int]:
    try:
        recording_id, utterance_idx = rollout_path.stem.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot parse TED-LIUM rollout stem: {rollout_path.stem}") from exc

    split = rollout_path.parent.name
    sph_path = tedlium_base / split / "sph" / f"{recording_id}.sph"
    stm_path = tedlium_base / split / "stm" / f"{recording_id}.stm"
    return sph_path, stm_path, int(utterance_idx)


def load_waveform_segment(sph_path: Path, start_s: float, end_s: float) -> tuple[torch.Tensor, int]:
    info = torchaudio.info(str(sph_path))
    frame_offset = max(0, int(round(start_s * info.sample_rate)))
    num_frames = max(1, int(round((end_s - start_s) * info.sample_rate)))
    waveform, sample_rate = torchaudio.load(str(sph_path), frame_offset=frame_offset, num_frames=num_frames)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def load_ssl_bundle(name: str):
    try:
        bundle = getattr(torchaudio.pipelines, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torchaudio SSL bundle: {name}") from exc
    return bundle, bundle.get_model().eval()


@torch.no_grad()
def extract_features(model, waveform, lengths):
    extracted = model.extract_features(waveform, lengths=lengths)
    if isinstance(extracted, tuple):
        features, feature_lengths = extracted
    else:
        features, feature_lengths = extracted, None
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if feature_lengths is None:
        feature_lengths = torch.tensor([features.size(1)], dtype=torch.long, device=features.device)
    return features, feature_lengths


def align_features(features: torch.Tensor, target_steps: int) -> torch.Tensor:
    features = features.transpose(1, 2)
    features = F.interpolate(features, size=target_steps, mode="linear", align_corners=False)
    return features.transpose(1, 2).squeeze(0).contiguous()


def build_one(
    rollout_path: Path,
    output_path: Path,
    tedlium_base: Path,
    bundle,
    model,
    device: torch.device,
    overwrite: bool,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        cached = torch.load(output_path, map_location="cpu", weights_only=False)
        return {
            "status": "cached",
            "rollout": str(rollout_path),
            "output": str(output_path),
            "shape": list(cached["ssl_features"].shape),
        }

    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    target_steps = int(rollout["generation"].shape[-1])
    expected_frames = int(rollout["audio"].shape[-1])

    sph_path, stm_path, utterance_idx = rollout_to_tedlium_paths(rollout_path, tedlium_base)
    utterances = parse_stm(stm_path)
    if utterance_idx >= len(utterances):
        raise IndexError(f"{rollout_path} maps to utterance {utterance_idx}, but {stm_path} has {len(utterances)} utterances")
    utterance = utterances[utterance_idx]

    waveform, sample_rate = load_waveform_segment(sph_path, utterance["start"], utterance["end"])
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate

    waveform = waveform.to(device)
    lengths = torch.tensor([waveform.size(-1)], dtype=torch.long, device=device)
    features, feature_lengths = extract_features(model, waveform, lengths)
    features = features[:, : int(feature_lengths[0].item())]
    aligned = align_features(features.cpu(), target_steps).to(dtype=torch.float16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ssl_features": aligned,
            "ssl_bundle": bundle.__class__.__name__,
            "rollout_path": str(rollout_path),
            "sph_path": str(sph_path),
            "stm_path": str(stm_path),
            "utterance_idx": utterance_idx,
            "start": utterance["start"],
            "end": utterance["end"],
            "sample_rate": sample_rate,
            "target_steps": target_steps,
            "expected_spectrogram_frames": expected_frames,
        },
        output_path,
    )
    return {
        "status": "built",
        "rollout": str(rollout_path),
        "output": str(output_path),
        "shape": list(aligned.shape),
        "seconds": float(utterance["end"]) - float(utterance["start"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=Path("/store/store4/data/l2augment_rollout_uvqmlm"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tedlium-base", type=Path, default=Path("/store/store4/data/TEDLIUM_release-3/legacy"))
    parser.add_argument("--split", choices=["train", "dev"], action="append", required=True)
    parser.add_argument("--ssl-bundle", default="HUBERT_BASE")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    bundle, model = load_ssl_bundle(args.ssl_bundle)
    device = torch.device(args.device)
    model = model.to(device)

    results = []
    for split in args.split:
        rollout_paths = sorted((args.rollout_root / split).glob("*.pt"))
        if args.max_files is not None:
            rollout_paths = rollout_paths[: args.max_files]
        for i, rollout_path in enumerate(rollout_paths, start=1):
            output_path = args.cache_root / split / rollout_path.name
            result = build_one(
                rollout_path=rollout_path,
                output_path=output_path,
                tedlium_base=args.tedlium_base,
                bundle=bundle,
                model=model,
                device=device,
                overwrite=args.overwrite,
            )
            results.append(result)
            print(f"[{split} {i}/{len(rollout_paths)}] {result['status']} {rollout_path.name} -> {result['shape']}")

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
