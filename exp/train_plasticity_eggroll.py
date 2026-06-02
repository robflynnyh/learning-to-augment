import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional
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


def build_checkpoint_payload(updater, optimizer, config, step: int, metrics: dict) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "checkpoint_type": "plasticity_updater_only",
        "contains_asr_model_state": False,
        "updater_state_dict": updater.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
        "step": step,
        "metrics": metrics,
    }
    if bool(config.get("training", {}).get("save_optimizer_state", True)):
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def _checkpoint_paths(config, step: int) -> Tuple[Path, Optional[Path]]:
    training_cfg = config.get("training", {})
    checkpoint_dir = training_cfg.get("checkpoint_dir")
    model_save_path = training_cfg.get("model_save_path")
    if checkpoint_dir:
        root = Path(checkpoint_dir)
        step_path = root / f"updater_step_{step:08d}.pt"
        latest_path = Path(model_save_path) if model_save_path else root / "latest.pt"
        return step_path, latest_path
    if not model_save_path:
        raise ValueError("training.model_save_path or training.checkpoint_dir is required")
    return Path(model_save_path), None


def _prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("updater_step_*.pt"))
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def save_checkpoint(updater, optimizer, config, step: int, metrics: dict) -> Path:
    save_path, latest_path = _checkpoint_paths(config, step)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(updater, optimizer, config, step, metrics)
    torch.save(payload, save_path)

    if latest_path is not None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, latest_path)
        keep_last = int(config.get("training", {}).get("keep_last_checkpoints", 3))
        _prune_old_checkpoints(save_path.parent, keep_last)
    return save_path


def load_updater_checkpoint(updater, optimizer, path: str, device) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("updater_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise KeyError(f"Checkpoint {path} does not contain updater_state_dict")
    updater.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("step", 0))


def init_wandb(config):
    training_cfg = config.get("training", {})
    enabled = bool(training_cfg.get("wandb_enabled", False))
    if not enabled:
        return None

    import wandb

    kwargs = {
        "project": training_cfg.get("wandb_project", "l2augment"),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    optional_fields = {
        "name": "wandb_name",
        "mode": "wandb_mode",
        "id": "wandb_id",
        "resume": "wandb_resume",
        "group": "wandb_group",
        "dir": "wandb_dir",
    }
    for wandb_key, config_key in optional_fields.items():
        value = training_cfg.get(config_key)
        if value is not None:
            kwargs[wandb_key] = value
    return wandb.init(**kwargs)


def log_metrics(wandb_run, metrics: dict) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=int(metrics["step"]))


def merge_dotlist_overrides(config, overrides: Sequence[str]):
    if not overrides:
        return config
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def _metric_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _fast_weight_rank_mean(info: dict) -> float:
    ranks = info.get("final_fast_state_rank")
    if ranks is None:
        return 0.0
    return _metric_float(ranks.float().mean())


def _fast_weight_rank_max(info: dict) -> float:
    ranks = info.get("final_fast_state_rank")
    if ranks is None:
        return 0.0
    return _metric_float(ranks.float().max())


def _print_metrics(metrics: dict) -> None:
    print(json.dumps(metrics, sort_keys=True), flush=True)


def _resolve_resume_path(config) -> Optional[str]:
    training_cfg = config.get("training", {})
    if training_cfg.get("resume_model_path"):
        return str(training_cfg["resume_model_path"])
    if not training_cfg.get("resume", False):
        return None
    if training_cfg.get("model_save_path") and os.path.exists(training_cfg["model_save_path"]):
        return str(training_cfg["model_save_path"])
    checkpoint_dir = training_cfg.get("checkpoint_dir")
    if checkpoint_dir:
        latest_path = Path(checkpoint_dir) / "latest.pt"
        if latest_path.exists():
            return str(latest_path)
    return None


def assert_asr_frozen(asr_model) -> None:
    trainable = [name for name, param in asr_model.named_parameters() if param.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:5])
        raise RuntimeError(f"ASR model must remain frozen; trainable params include: {preview}")


def adaptation_parameter_count(module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def write_run_summary(config, metrics: dict, checkpoint_path: Optional[Path]) -> None:
    summary_path = config.get("training", {}).get("summary_path")
    if not summary_path:
        return
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_text = str(checkpoint_path) if checkpoint_path is not None else "not yet saved"
    path.write_text(
        "\n".join(
            [
                "# ROB-186 Plasticity EGGROLL Run",
                "",
                f"- Last step: `{metrics['step']}`",
                f"- Last checkpoint: `{checkpoint_text}`",
                f"- Mean WER: `{metrics['wer_mean']:.6f}`",
                f"- Mean quality: `{metrics['quality_mean']:.6f}`",
                f"- Reward std: `{metrics['reward_std']:.6f}`",
                "- Checkpoint payload type: `plasticity_updater_only`",
                "- Seed ASR checkpoint is loaded as frozen context and is not saved in updater checkpoints.",
            ]
        )
        + "\n",
        encoding="utf-8",
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
    assert_asr_frozen(asr_model)

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

    optimizer = build_optimizer(updater.parameters(), config)
    resume_path = _resolve_resume_path(config)
    start_step = 0
    if resume_path is not None:
        start_step = load_updater_checkpoint(updater, optimizer, resume_path, device)

    wandb_run = init_wandb(config)
    print(
        json.dumps(
            {
                "event": "plasticity_train_start",
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "target_modules": target_modules,
                "adaptation_parameters": adaptation_parameter_count(updater),
                "checkpoint_type": "plasticity_updater_only",
                "resume_step": start_step,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    data = dataset_functions[config["training"].get("dataset", "tedlium3_segmented_data")](
        config["training"].get("split", "train")
    )

    num_steps = int(config["training"].get("num_steps", 1))
    log_every = int(config["training"].get("log_every", 10))
    checkpoint_every = int(config["training"].get("checkpoint_every", 500))
    batch_size = int(config["rollout"].get("batch_size_recordings", 1))
    checkpoint_path = None

    try:
        for step in range(start_step + 1, start_step + num_steps + 1):
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
                "wer_mean": _metric_float(info["wer"].mean()),
                "quality_mean": _metric_float(info["quality"].mean()),
                "reward_std": _metric_float(rewards.std(unbiased=False)),
                "reward_mean": _metric_float(rewards.mean()),
                "fast_weight_rank_mean": _fast_weight_rank_mean(info),
                "fast_weight_rank_max": _fast_weight_rank_max(info),
                "adaptation_parameters": adaptation_parameter_count(updater),
            }
            if step % log_every == 0 or step == start_step + 1:
                _print_metrics(metrics)
            log_metrics(wandb_run, metrics)
            if step % checkpoint_every == 0 or step == start_step + num_steps:
                checkpoint_path = save_checkpoint(updater, optimizer, config, step, metrics)
                write_run_summary(config, metrics, checkpoint_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, for example --set training.num_steps=1",
    )
    args = parser.parse_args()
    main(merge_dotlist_overrides(OmegaConf.load(args.config), args.overrides))
