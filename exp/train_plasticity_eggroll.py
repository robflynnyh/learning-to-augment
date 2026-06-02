import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from omegaconf import OmegaConf

from l2augment.modelling.plasticity import (
    PlasticityPolicy,
    collect_linear_specs,
    wrap_linear_modules,
)
from l2augment.rollout.gpu_plasticity import rollout_recordings_with_plasticity_candidates
from l2augment.utils.data import dataset_functions
from l2augment.utils.eggroll import eggroll_update_shared_params, sample_rank1_perturbations


def load_asr_from_config(config, tokenizer, device, dtype):
    from lcasr.utils.general import get_model_class
    from lcasr.utils.general import load_model as load_asr_model

    checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
    asr_model_class = get_model_class(config=config)
    asr_model = load_asr_model(
        checkpoint["config"],
        tokenizer.vocab_size(),
        asr_model_class,
    )
    asr_model.load_state_dict(checkpoint["model"])
    if hasattr(asr_model, "flash_attn"):
        asr_model.flash_attn = False
    asr_model.to(device=device, dtype=dtype)
    asr_model.eval()
    for param in asr_model.parameters():
        param.requires_grad_(False)
    return asr_model


def normalise_processed_recording(processed) -> Tuple[torch.Tensor, str]:
    if isinstance(processed, tuple) and len(processed) == 2:
        audio, text = processed
    elif isinstance(processed, dict):
        audio = processed.get("spectrogram", processed.get("audio"))
        text = processed.get("text", processed.get("transcript"))
    elif isinstance(processed, list):
        audio_parts, text_parts = [], []
        for item in processed:
            audio_part = item.get("spectrogram", item.get("audio"))
            text_part = item.get("text", item.get("transcript", ""))
            audio_parts.append(audio_part.squeeze(0) if audio_part.dim() == 3 else audio_part)
            text_parts.append(text_part)
        audio = torch.cat(audio_parts, dim=-1)
        text = " ".join(text_parts)
    else:
        raise TypeError(f"Unsupported processed recording type: {type(processed)!r}")

    if audio.dim() == 3 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.dim() != 2:
        raise ValueError(f"Expected recording audio [C,S], got {tuple(audio.shape)}")
    return audio, str(text)


def sample_labelled_batch(data, batch_size: int) -> Tuple[torch.Tensor, List[str]]:
    indexes = random.sample(range(len(data)), k=min(batch_size, len(data)))
    audios, texts = [], []
    for idx in indexes:
        item = data[idx]
        audio, text = normalise_processed_recording(item["process_fn"](item))
        audios.append(audio)
        texts.append(text)

    channels = audios[0].shape[0]
    max_len = max(audio.shape[-1] for audio in audios)
    batch = torch.zeros(len(audios), channels, max_len, dtype=audios[0].dtype)
    for bidx, audio in enumerate(audios):
        if audio.shape[0] != channels:
            raise ValueError("All recordings in a batch must have the same feature count")
        batch[bidx, :, : audio.shape[-1]] = audio
    return batch, texts


def build_optimizer(parameters, config):
    name = str(config.get("eggroll", {}).get("optimizer", "adam")).lower()
    lr = float(config.get("eggroll", {}).get("lr", 1e-4))
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def save_checkpoint(updater, optimizer, config, step: int, metrics: dict) -> None:
    save_path = Path(config["training"]["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": updater.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
            "step": step,
            "metrics": metrics,
        },
        save_path,
    )


def main(config):
    from lcasr.utils.audio_tools import load_tokenizer

    seed = int(config.get("training", {}).get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(config.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, config.get("training", {}).get("dtype", "float32"))
    tokenizer = load_tokenizer()
    asr_model = load_asr_from_config(config, tokenizer, device, dtype)

    target_modules = list(config["plasticity"]["target_modules"])
    module_specs = wrap_linear_modules(asr_model, target_modules)
    module_specs = collect_linear_specs(asr_model, target_modules)

    updater = PlasticityPolicy(
        module_specs,
        token_dim=int(config["plasticity"].get("token_dim", 128)),
        comm_dim=int(config["plasticity"].get("comm_dim", 128)),
        update_rank=int(config["plasticity"].get("update_rank", 1)),
        max_eta=float(config["plasticity"].get("max_eta", 1e-4)),
        default_rho=float(config["plasticity"].get("default_rho", 0.95)),
    ).to(device=device, dtype=dtype)

    checkpoint_path = config.get("training", {}).get("model_save_path")
    if checkpoint_path and os.path.exists(checkpoint_path) and config.get("training", {}).get("resume", False):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        updater.load_state_dict(checkpoint["model_state_dict"])

    optimizer = build_optimizer(updater.parameters(), config)
    data = dataset_functions[config["training"].get("dataset", "tedlium3_segmented_data")](
        config["training"].get("split", "train")
    )

    num_steps = int(config["training"].get("num_steps", 1))
    log_every = int(config["training"].get("log_every", 10))
    checkpoint_every = int(config["training"].get("checkpoint_every", 500))
    batch_size = int(config["rollout"].get("batch_size_recordings", 1))

    for step in range(1, num_steps + 1):
        audio_batch, reference_text_batch = sample_labelled_batch(data, batch_size)
        audio_batch = audio_batch.to(device=device, dtype=dtype)
        perturbations = sample_rank1_perturbations(
            updater,
            int(config["eggroll"].get("num_candidates", 8)),
            antithetic=bool(config["eggroll"].get("antithetic", True)),
            device=device,
            dtype=dtype,
        )
        rewards, info = rollout_recordings_with_plasticity_candidates(
            asr_model=asr_model,
            updater=updater,
            recording_audio_batch=audio_batch,
            reference_text_batch=reference_text_batch,
            tokenizer=tokenizer,
            candidate_perturbations=perturbations,
            config=config,
            module_specs=module_specs,
        )
        eggroll_update_shared_params(
            updater,
            perturbations,
            rewards,
            float(config["eggroll"].get("sigma", 1e-3)),
            optimizer,
        )

        metrics = {
            "step": step,
            "wer_mean": float(info["wer"].mean().detach().cpu()),
            "quality_mean": float(info["quality"].mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
        }
        if step % log_every == 0 or step == 1:
            print(json.dumps(metrics, sort_keys=True))
        if step % checkpoint_every == 0 or step == num_steps:
            save_checkpoint(updater, optimizer, config, step, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True)
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
