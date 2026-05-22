#!/usr/bin/env python3
import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in SCRIPT_PATH.parents if (parent / "l2augment" / "modelling" / "models.py").exists())
sys.path.insert(0, str(REPO_ROOT))

from l2augment.modelling.models import RewardConditionedMaskLM
from l2augment.rollout.cpu_multistep_oracle import cpu_rollout_policy
from l2augment.utils.data import tedlium3_segmented_data
from l2augment.utils.helpers import load_asr_model_fn
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import get_model_class, load_model as load_asr_model


DEFAULT_RECORDING_IDS = [
    "AlGore_2009",
    "BarrySchwartz_2005G",
    "BlaiseAguerayArcas_2007",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(config, checkpoint_path):
    policy_config = OmegaConf.to_container(config.policy.config, resolve=True)
    policy = RewardConditionedMaskLM(**policy_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint


def load_asr_components(config):
    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config=config)
    checkpoint = torch.load(config.checkpointing.asr_model, map_location="cpu", weights_only=False)
    asr_model_config = checkpoint["config"]
    asr_model_state_dict = checkpoint["model"]
    loader = partial(
        load_asr_model_fn,
        load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
        asr_model_state_dict,
    )
    return tokenizer, asr_model_config, asr_model_class, loader


def resolve_recordings(split, base_path, recording_ids):
    dataset = tedlium3_segmented_data()(split, base_path=base_path)
    by_id = {item["id"]: (index, item) for index, item in enumerate(dataset)}
    missing = [recording_id for recording_id in recording_ids if recording_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing TED-LIUM {split} recordings: {missing}")
    return [{"dataset_index": by_id[recording_id][0], **by_id[recording_id][1]} for recording_id in recording_ids]


def summarize_run(recording, reward, rollout_output):
    original_wer = float(rollout_output["original_wer"])
    updated_wer = float(rollout_output["updated_wer"])
    return {
        "recording_id": recording["id"],
        "dataset_index": int(recording["dataset_index"]),
        "conditioning_reward": float(reward),
        "num_utterances": int(len(rollout_output["reference"])),
        "original_wer": original_wer,
        "updated_wer": updated_wer,
        "wer_delta_updated_minus_original": updated_wer - original_wer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml")
    parser.add_argument("--checkpoint", default="/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt")
    parser.add_argument("--tedlium-base", default="/store/store4/data/TEDLIUM_release-3/legacy/")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--reward", type=float, action="append", default=[0.0, 1.0])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--output", default="exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_adaptation_wer_reward_0_vs_1.json")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    policy, policy_checkpoint = build_policy(config, args.checkpoint)
    tokenizer, asr_model_config, asr_model_class, asr_loader = load_asr_components(config)
    recording_ids = args.recording_id or DEFAULT_RECORDING_IDS
    recordings = resolve_recordings(args.split, args.tedlium_base, recording_ids)

    results = []
    for recording_index, recording in enumerate(recordings):
        utterances = recording["process_fn"](recording)
        for reward_index, reward in enumerate(args.reward):
            run_seed = args.seed + recording_index * 100 + reward_index
            seed_all(run_seed)
            print(
                f"[rob117-wer] {recording['id']} reward={reward} "
                f"utterances={len(utterances)} seed={run_seed}",
                flush=True,
            )
            rollout_output = cpu_rollout_policy(
                policy=policy,
                load_asr_model_fn=asr_loader,
                tokenizer=tokenizer,
                utterances=utterances,
                asr_model_config=asr_model_config,
                asr_model_class=asr_model_class,
                augmentation_config={"conditioning_reward": reward, "sample": args.sample},
                epochs=args.epochs,
                optim_args={"lr": args.lr},
                shuffle_seed=run_seed,
            )
            results.append(summarize_run(recording, reward, rollout_output))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_bytes": int(Path(args.checkpoint).stat().st_size),
        "policy_checkpoint_config_id": policy_checkpoint.get("config", {}).get("evaluation", {}).get("id"),
        "tedlium_base": args.tedlium_base,
        "split": args.split,
        "recording_ids": recording_ids,
        "rewards": [float(reward) for reward in args.reward],
        "sample": bool(args.sample),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
