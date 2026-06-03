import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf

from exp.train_plasticity_eggroll import (
    load_asr_from_config,
    load_updater_checkpoint,
    merge_dotlist_overrides,
    normalise_processed_recording,
    parse_training_dtype,
    resolve_target_modules,
)
from l2augment.modelling.plasticity import PlasticityPolicy, wrap_linear_modules
from l2augment.rollout.gpu_plasticity import rollout_recordings_with_plasticity_candidates
from l2augment.utils.data import dataset_functions
from l2augment.utils.eggroll import EggrollPerturbations, Rank1Perturbation
from l2augment.utils.eggroll import iter_eggroll_linears


def build_eval_batch(data, indexes: Sequence[int]) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str]]:
    ids, audios, texts = [], [], []
    for idx in indexes:
        item = data[idx]
        audio, text = normalise_processed_recording(item["process_fn"](item))
        ids.append(str(item.get("id", idx)))
        audios.append(audio)
        texts.append(text)

    channels = audios[0].shape[0]
    max_len = max(audio.shape[-1] for audio in audios)
    batch = torch.zeros(len(audios), channels, max_len, dtype=audios[0].dtype)
    for bidx, audio in enumerate(audios):
        if audio.shape[0] != channels:
            raise ValueError("All recordings in an eval batch must have the same feature count")
        batch[bidx, :, : audio.shape[-1]] = audio
    lengths = torch.tensor([audio.shape[-1] for audio in audios], dtype=torch.long)
    return ids, batch, lengths, texts


def zero_center_perturbations(updater: torch.nn.Module, device, dtype) -> EggrollPerturbations:
    layers: Dict[str, Rank1Perturbation] = {}
    for name, module in iter_eggroll_linears(updater):
        out_features, in_features = module.weight.shape
        layers[name] = Rank1Perturbation(
            a=torch.zeros(1, out_features, device=device, dtype=dtype),
            b=torch.zeros(1, in_features, device=device, dtype=dtype),
            antithetic=False,
        )
    return EggrollPerturbations(layers)


def summarise_eval_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"num_recordings": 0, "wer_mean": None, "quality_mean": None}
    wer_values = [float(row["wer"]) for row in rows]
    quality_values = [float(row["quality"]) for row in rows]
    return {
        "num_recordings": len(rows),
        "wer_mean": sum(wer_values) / len(wer_values),
        "wer_min": min(wer_values),
        "wer_max": max(wer_values),
        "quality_mean": sum(quality_values) / len(quality_values),
    }


def resolve_eval_indexes(eval_cfg, dataset_size: int) -> List[int]:
    start_index = int(eval_cfg.get("start_index", 0))
    if start_index < 0 or start_index > dataset_size:
        raise ValueError(f"evaluation.start_index={start_index} is outside dataset size {dataset_size}")

    requested = eval_cfg.get("num_recordings", 3)
    if isinstance(requested, str) and requested.lower() == "all":
        stop_index = dataset_size
    else:
        count = int(requested)
        if count < 0:
            raise ValueError("evaluation.num_recordings must be non-negative or 'all'")
        stop_index = min(start_index + count, dataset_size)
    return list(range(start_index, stop_index))


def resolve_eval_variants(eval_cfg, latest_checkpoint: Optional[str]) -> List[Tuple[str, Optional[str], bool]]:
    requested = eval_cfg.get("variants")
    if requested is None:
        variants = ["step0_random_init"]
        if latest_checkpoint:
            variants.append("latest_checkpoint")
    elif isinstance(requested, str):
        variants = [part.strip() for part in requested.split(",") if part.strip()]
    else:
        variants = [str(part) for part in requested]

    resolved = []
    for variant in variants:
        if variant == "seed_asr":
            resolved.append((variant, None, True))
        elif variant == "step0_random_init":
            resolved.append((variant, None, False))
        elif variant == "latest_checkpoint":
            if not latest_checkpoint:
                raise ValueError("latest_checkpoint variant requested but no checkpoint path was configured")
            resolved.append((variant, str(latest_checkpoint), False))
        else:
            raise ValueError(
                "Unknown evaluation variant "
                f"{variant!r}; expected seed_asr, step0_random_init, or latest_checkpoint"
            )
    return resolved


def evaluate_variant(
    *,
    variant: str,
    checkpoint_path: Optional[str],
    seed_asr: bool = False,
    config,
    asr_model,
    module_specs,
    tokenizer,
    audio_batch: torch.Tensor,
    recording_lengths: torch.Tensor,
    reference_texts: Sequence[str],
    recording_ids: Sequence[str],
    recording_indexes: Optional[Sequence[int]] = None,
    device,
    dtype,
) -> Tuple[List[dict], dict]:
    updater = PlasticityPolicy(
        module_specs,
        token_dim=int(config["plasticity"].get("token_dim", 128)),
        comm_dim=int(config["plasticity"].get("comm_dim", 128)),
        update_rank=int(config["plasticity"].get("update_rank", 1)),
        max_eta=0.0 if seed_asr else float(config["plasticity"].get("max_eta", 1e-4)),
        default_rho=float(config["plasticity"].get("default_rho", 0.95)),
    ).to(device=device, dtype=dtype)
    checkpoint_step = 0
    if checkpoint_path is not None:
        checkpoint_step = load_updater_checkpoint(updater, optimizer=None, path=checkpoint_path, device=device)
    perturbations = zero_center_perturbations(updater, device=device, dtype=dtype)

    _, info = rollout_recordings_with_plasticity_candidates(
        asr_model=asr_model,
        updater=updater,
        recording_audio_batch=audio_batch,
        recording_lengths=recording_lengths,
        reference_text_batch=reference_texts,
        tokenizer=tokenizer,
        candidate_perturbations=perturbations,
        config=config,
        module_specs=module_specs,
    )

    wer = info["wer"].detach().float().cpu()[:, 0]
    quality = info["quality"].detach().float().cpu()[:, 0]
    chunks = info["chunks_per_recording"].detach().cpu()
    rows = []
    for idx, rec_id in enumerate(recording_ids):
        rows.append(
            {
                "variant": variant,
                "checkpoint_path": checkpoint_path or "",
                "checkpoint_step": checkpoint_step,
                "recording_index": int(recording_indexes[idx]) if recording_indexes is not None else idx,
                "recording_id": rec_id,
                "chunks": int(chunks[idx]),
                "wer": float(wer[idx]),
                "quality": float(quality[idx]),
            }
        )
    summary = summarise_eval_rows(rows)
    summary.update(
        {
            "variant": variant,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": checkpoint_step,
            "fast_state_norm_ratio_mean": float(info["fast_state_norm_ratio_mean"].detach().float().cpu()),
            "fast_state_norm_ratio_max": float(info["fast_state_norm_ratio_max"].detach().float().cpu()),
            "fast_weight_clipped_fraction": float(info["fast_weight_clipped_fraction"].detach().float().cpu()),
        }
    )
    return rows, summary


def write_outputs(result_dir: Path, rows: Sequence[dict], summaries: Sequence[dict], config) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    rows_path = result_dir / "per_recording.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "checkpoint_path",
                "checkpoint_step",
                "recording_index",
                "recording_id",
                "chunks",
                "wer",
                "quality",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(list(summaries), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path = result_dir / "resolved_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

    lines = [
        "# ROB-186 Plasticity Test Evaluation",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Per-recording CSV: `{rows_path}`",
        f"- Summary JSON: `{summary_path}`",
        "",
        "| Variant | Step | Recordings | Mean WER | Mean quality | Fast-state max | Clipped fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(value, digits=6):
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    for summary in summaries:
        lines.append(
            "| {variant} | {checkpoint_step} | {num_recordings} | {wer_mean} | "
            "{quality_mean} | {fast_state_norm_ratio_max} | "
            "{fast_weight_clipped_fraction} |".format(
                variant=summary["variant"],
                checkpoint_step=summary["checkpoint_step"],
                num_recordings=summary["num_recordings"],
                wer_mean=fmt(summary["wer_mean"]),
                quality_mean=fmt(summary["quality_mean"]),
                fast_state_norm_ratio_max=fmt(summary["fast_state_norm_ratio_max"]),
                fast_weight_clipped_fraction=fmt(summary["fast_weight_clipped_fraction"]),
            )
        )
    (result_dir / "OUTCOME.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config) -> None:
    from lcasr.utils.audio_tools import load_tokenizer

    seed = int(config.get("evaluation", {}).get("seed", config.get("training", {}).get("seed", 0)))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(config.get("evaluation", {}).get("device", config.get("training", {}).get("device", "cuda")))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    dtype = parse_training_dtype(config)
    tokenizer = load_tokenizer()
    asr_model = load_asr_from_config(config, tokenizer, device, dtype)
    target_modules = resolve_target_modules(asr_model, config)
    module_specs = wrap_linear_modules(asr_model, target_modules)

    eval_cfg = config.get("evaluation", {})
    dataset_name = str(eval_cfg.get("dataset", config.get("training", {}).get("dataset", "tedlium")))
    split = str(eval_cfg.get("split", "test"))
    data = dataset_functions[dataset_name](split)
    indexes = resolve_eval_indexes(eval_cfg, len(data))
    batch_size = int(eval_cfg.get("batch_size_recordings", config.get("rollout", {}).get("batch_size_recordings", 1)))
    if batch_size <= 0:
        raise ValueError("evaluation.batch_size_recordings must be positive")

    latest_checkpoint = eval_cfg.get("latest_checkpoint", config.get("training", {}).get("model_save_path"))
    variants = resolve_eval_variants(eval_cfg, latest_checkpoint)

    all_rows, summaries = [], []
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "split": split,
                "num_dataset_recordings": len(data),
                "num_eval_recordings": len(indexes),
                "batch_size_recordings": batch_size,
                "variants": [variant for variant, _, _ in variants],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for variant, checkpoint_path, seed_asr in variants:
        variant_rows = []
        batch_summaries = []
        for batch_start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[batch_start : batch_start + batch_size]
            recording_ids, audio_batch, recording_lengths, reference_texts = build_eval_batch(data, batch_indexes)
            audio_batch = audio_batch.to(device=device, dtype=dtype)
            recording_lengths = recording_lengths.to(device=device)
            rows, batch_summary = evaluate_variant(
                variant=variant,
                checkpoint_path=checkpoint_path,
                seed_asr=seed_asr,
                config=config,
                asr_model=asr_model,
                module_specs=module_specs,
                tokenizer=tokenizer,
                audio_batch=audio_batch,
                recording_lengths=recording_lengths,
                reference_texts=reference_texts,
                recording_ids=recording_ids,
                recording_indexes=batch_indexes,
                device=device,
                dtype=dtype,
            )
            variant_rows.extend(rows)
            batch_summaries.append(batch_summary)
            print(
                json.dumps(
                    {
                        "event": "eval_batch",
                        "variant": variant,
                        "batch_start": batch_start,
                        "batch_size": len(batch_indexes),
                        "wer_mean_so_far": summarise_eval_rows(variant_rows)["wer_mean"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        summary = summarise_eval_rows(variant_rows)
        fast_max_values = [
            float(batch_summary["fast_state_norm_ratio_max"])
            for batch_summary in batch_summaries
            if batch_summary.get("fast_state_norm_ratio_max") is not None
        ]
        clipped_values = [
            float(batch_summary["fast_weight_clipped_fraction"])
            for batch_summary in batch_summaries
            if batch_summary.get("fast_weight_clipped_fraction") is not None
        ]
        summary.update(
            {
                "variant": variant,
                "checkpoint_path": checkpoint_path,
                "checkpoint_step": batch_summaries[0]["checkpoint_step"] if batch_summaries else 0,
                "fast_state_norm_ratio_mean": (
                    sum(float(batch_summary["fast_state_norm_ratio_mean"]) for batch_summary in batch_summaries)
                    / len(batch_summaries)
                    if batch_summaries
                    else None
                ),
                "fast_state_norm_ratio_max": max(fast_max_values) if fast_max_values else None,
                "fast_weight_clipped_fraction": (
                    sum(clipped_values) / len(clipped_values) if clipped_values else None
                ),
            }
        )
        all_rows.extend(variant_rows)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    result_dir = Path(eval_cfg.get("result_dir", "exp/results/plasticity_eggroll/test_eval"))
    write_outputs(result_dir, all_rows, summaries, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, for example --set evaluation.num_recordings=3",
    )
    args = parser.parse_args()
    main(merge_dotlist_overrides(OmegaConf.load(args.config), args.overrides))
