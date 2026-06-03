from pathlib import Path

import torch
from omegaconf import OmegaConf

from exp.train_plasticity_eggroll import (
    build_optimizer,
    build_checkpoint_payload,
    combine_rollout_infos,
    load_updater_checkpoint,
    merge_dotlist_overrides,
    parse_training_dtype,
    parse_rollout_devices,
    save_checkpoint,
    validate_training_dataset,
)
from exp.eval_plasticity_eggroll import summarise_eval_rows, zero_center_perturbations
from l2augment.utils.eggroll import EggrollLinear
from l2augment.utils.eggroll import group_normalise_rewards


def _config(root: Path):
    return OmegaConf.create(
        {
            "checkpointing": {"asr_model": "/store/example/frozen_asr.pt"},
            "training": {
                "checkpoint_dir": str(root / "checkpoints"),
                "model_save_path": str(root / "checkpoints" / "latest.pt"),
                "keep_last_checkpoints": 2,
                "save_optimizer_state": True,
            },
        }
    )


def test_plasticity_checkpoint_payload_is_updater_only(tmp_path):
    updater = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(updater.parameters(), lr=0.1)
    payload = build_checkpoint_payload(updater, optimizer, _config(tmp_path), 7, {"wer_mean": 0.5})

    assert payload["checkpoint_type"] == "plasticity_updater_only"
    assert payload["contains_asr_model_state"] is False
    assert "updater_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "model_state_dict" not in payload
    assert "asr_model_state_dict" not in payload


def test_plasticity_checkpoint_retention_keeps_latest_and_bounded_steps(tmp_path):
    updater = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(updater.parameters(), lr=0.1)
    config = _config(tmp_path)

    for step in range(1, 4):
        save_checkpoint(updater, optimizer, config, step, {"step": step})

    checkpoint_dir = Path(config.training.checkpoint_dir)
    step_names = sorted(path.name for path in checkpoint_dir.glob("updater_step_*.pt"))
    assert step_names == ["updater_step_00000002.pt", "updater_step_00000003.pt"]
    assert (checkpoint_dir / "latest.pt").is_file()

    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["checkpoint_type"] == "plasticity_updater_only"
    assert latest["step"] == 3


def test_load_updater_checkpoint_accepts_new_payload(tmp_path):
    updater = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(updater.parameters(), lr=0.1)
    config = _config(tmp_path)
    save_path = save_checkpoint(updater, optimizer, config, 5, {"step": 5})

    restored = torch.nn.Linear(2, 3)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    loaded_step = load_updater_checkpoint(restored, restored_optimizer, str(save_path), "cpu")

    assert loaded_step == 5
    for key, value in updater.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_train_script_dotlist_overrides():
    config = OmegaConf.create({"training": {"num_steps": 10}, "eggroll": {"num_candidates": 8}})
    merged = merge_dotlist_overrides(
        config,
        ["training.num_steps=1", "eggroll.num_candidates=2", "training.wandb_mode=offline"],
    )

    assert merged.training.num_steps == 1
    assert merged.eggroll.num_candidates == 2
    assert merged.training.wandb_mode == "offline"


def test_train_script_parses_bfloat16_dtype_aliases():
    assert parse_training_dtype(OmegaConf.create({"training": {"dtype": "bfloat16"}})) is torch.bfloat16
    assert parse_training_dtype(OmegaConf.create({"training": {"dtype": "bf16"}})) is torch.bfloat16


def test_train_script_rejects_unknown_dtype():
    config = OmegaConf.create({"training": {"dtype": "float8"}})

    try:
        parse_training_dtype(config)
    except ValueError as exc:
        assert "Unsupported training.dtype" in str(exc)
    else:
        raise AssertionError("unknown dtype should be rejected")


def test_rollout_device_config_normalises_cuda_primary_without_duplicates():
    config = OmegaConf.create({"rollout": {"devices": ["cuda:0", "cuda:1"]}})

    devices = parse_rollout_devices(config, torch.device("cuda"))

    assert [str(device) for device in devices] == ["cuda:0", "cuda:1"]


def test_multi_device_rollout_info_combines_rewards_over_full_candidate_axis():
    shard_a = {
        "wer": torch.tensor([[0.0, 1.0], [0.5, 1.5]]),
        "chunks_per_recording": torch.tensor([2, 2]),
        "chunk_length_frames_mean": torch.tensor(2.0),
        "chunk_size_frames": torch.tensor(2.0),
        "chunk_overlap_frames": torch.tensor(0.0),
        "rollout_chunk_steps": torch.tensor(2.0),
        "rollout_streams": torch.tensor(4.0),
        "fast_state_norm_ratio_mean": torch.tensor(0.01),
        "fast_state_norm_ratio_max": torch.tensor(0.02),
        "fast_weight_clipped_fraction": torch.tensor(0.0),
        "fast_update_norm_ratio_mean": torch.tensor(0.001),
        "fast_update_norm_ratio_max": torch.tensor(0.002),
    }
    shard_b = {
        "wer": torch.tensor([[0.5, 1.5], [0.0, 1.0]]),
        "chunks_per_recording": torch.tensor([2, 2]),
        "chunk_length_frames_mean": torch.tensor(2.0),
        "chunk_size_frames": torch.tensor(2.0),
        "chunk_overlap_frames": torch.tensor(0.0),
        "rollout_chunk_steps": torch.tensor(2.0),
        "rollout_streams": torch.tensor(4.0),
        "fast_state_norm_ratio_mean": torch.tensor(0.03),
        "fast_state_norm_ratio_max": torch.tensor(0.04),
        "fast_weight_clipped_fraction": torch.tensor(0.5),
        "fast_update_norm_ratio_mean": torch.tensor(0.003),
        "fast_update_norm_ratio_max": torch.tensor(0.004),
    }

    rewards, info = combine_rollout_infos(
        [shard_a, shard_b],
        [torch.tensor([0, 2]), torch.tensor([1, 3])],
        num_candidates=4,
        device=torch.device("cpu"),
        reward_eps=1e-8,
    )

    expected_wer = torch.tensor([[0.0, 0.5, 1.0, 1.5], [0.5, 0.0, 1.5, 1.0]])
    expected_quality = 1.0 - expected_wer.clamp(max=1.0)
    expected_rewards_bn = group_normalise_rewards(expected_quality)
    torch.testing.assert_close(info["wer"], expected_wer)
    torch.testing.assert_close(info["quality"], expected_quality)
    torch.testing.assert_close(info["rewards_bn"], expected_rewards_bn)
    torch.testing.assert_close(rewards, expected_rewards_bn.mean(dim=0))
    assert info["rollout_num_devices"].item() == 2
    torch.testing.assert_close(info["rollout_streams_per_device"], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(info["fast_state_norm_ratio_mean"], torch.tensor(0.02))
    torch.testing.assert_close(info["fast_state_norm_ratio_max"], torch.tensor(0.04))
    torch.testing.assert_close(info["fast_weight_clipped_fraction"], torch.tensor(0.25))
    torch.testing.assert_close(info["fast_update_norm_ratio_mean"], torch.tensor(0.002))
    torch.testing.assert_close(info["fast_update_norm_ratio_max"], torch.tensor(0.004))


def test_train_script_supports_adamw_optimizer():
    model = torch.nn.Linear(2, 3)
    config = OmegaConf.create({"eggroll": {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.0}})

    optimizer = build_optimizer(model.parameters(), config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_plasticity_training_rejects_segmented_dataset_by_default():
    config = OmegaConf.create(
        {
            "training": {
                "dataset": "tedlium3_segmented_data",
                "require_long_form_recordings": True,
            }
        }
    )

    try:
        validate_training_dataset(config)
    except ValueError as exc:
        assert "requires unsegmented long-form recordings" in str(exc)
    else:
        raise AssertionError("segmented TED-LIUM loader should be rejected")


def test_plasticity_training_accepts_unsegmented_tedlium_dataset():
    config = OmegaConf.create(
        {
            "training": {
                "dataset": "tedlium",
                "require_long_form_recordings": True,
            }
        }
    )

    assert validate_training_dataset(config) == "tedlium"


def test_plasticity_eval_summary_averages_per_recording_rows():
    rows = [
        {"wer": 0.2, "quality": 0.8},
        {"wer": 0.6, "quality": 0.4},
    ]

    summary = summarise_eval_rows(rows)

    assert summary["num_recordings"] == 2
    assert summary["wer_mean"] == 0.4
    assert summary["quality_mean"] == 0.6000000000000001


def test_zero_center_perturbations_uses_single_candidate_zero_rank1_noise():
    module = torch.nn.Sequential(EggrollLinear(2, 3), torch.nn.LayerNorm(3))

    perturbations = zero_center_perturbations(module, device=torch.device("cpu"), dtype=torch.float32)

    assert perturbations.num_candidates == 1
    assert "0" in perturbations.layers
    torch.testing.assert_close(perturbations.get("0").a, torch.zeros(1, 3))
    torch.testing.assert_close(perturbations.get("0").b, torch.zeros(1, 2))
