from pathlib import Path

import torch
from omegaconf import OmegaConf

from exp.train_plasticity_eggroll import (
    build_checkpoint_payload,
    load_updater_checkpoint,
    merge_dotlist_overrides,
    save_checkpoint,
    validate_training_dataset,
)


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
