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


def evaluate_variant(
    *,
    variant: str,
    checkpoint_path: Optional[str],
    config,
    asr_model,
    module_specs,
    tokenizer,
    audio_batch: torch.Tensor,
    recording_lengths: torch.Tensor,
    reference_texts: Sequence[str],
    recording_ids: Sequence[str],
    device,
    dtype,
) -> Tuple[List[dict], dict]:
    updater = PlasticityPolicy(
        module_specs,
        token_dim=int(config["plasticity"].get("token_dim", 128)),
        comm_dim=int(config["plasticity"].get("comm_dim", 128)),
        update_rank=int(config["plasticity"].get("update_rank", 1)),
        max_eta=float(config["plasticity"].get("max_eta", 1e-4)),
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
                "recording_index": idx,
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
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
    for summary in summaries:
        lines.append(
            "| {variant} | {checkpoint_step} | {num_recordings} | {wer_mean:.6f} | "
            "{quality_mean:.6f} | {fast_state_norm_ratio_max:.6f} | "
            "{fast_weight_clipped_fraction:.6f} |".format(**summary)
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
    num_recordings = int(eval_cfg.get("num_recordings", 3))
    start_index = int(eval_cfg.get("start_index", 0))
    indexes = list(range(start_index, min(start_index + num_recordings, len(data))))
    recording_ids, audio_batch, recording_lengths, reference_texts = build_eval_batch(data, indexes)
    audio_batch = audio_batch.to(device=device, dtype=dtype)
    recording_lengths = recording_lengths.to(device=device)

    latest_checkpoint = eval_cfg.get("latest_checkpoint", config.get("training", {}).get("model_save_path"))
    variants = [("step0_random_init", None)]
    if latest_checkpoint:
        variants.append(("latest_checkpoint", str(latest_checkpoint)))

    all_rows, summaries = [], []
    for variant, checkpoint_path in variants:
        rows, summary = evaluate_variant(
            variant=variant,
            checkpoint_path=checkpoint_path,
            config=config,
            asr_model=asr_model,
            module_specs=module_specs,
            tokenizer=tokenizer,
            audio_batch=audio_batch,
            recording_lengths=recording_lengths,
            reference_texts=reference_texts,
            recording_ids=recording_ids,
            device=device,
            dtype=dtype,
        )
        all_rows.extend(rows)
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
