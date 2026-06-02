import copy
import inspect
from types import SimpleNamespace

import torch
from torch import nn

from l2augment.modelling import plasticity
from l2augment.modelling.plasticity import PlasticityPolicy, wrap_linear_modules
from l2augment.rollout import gpu_plasticity
from l2augment.rollout.gpu_plasticity import rollout_recordings_with_plasticity_candidates
from l2augment.utils.eggroll import (
    EggrollPerturbations,
    Rank1Perturbation,
    group_normalise_rewards,
    sample_rank1_perturbations,
)


class TinyASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapt = nn.Linear(2, 2, bias=False)

    def forward(self, audio_signal, length=None):
        del length
        x = audio_signal.transpose(1, 2)
        y = self.adapt(x)
        return {"final_posteriors": y}


def _config():
    return SimpleNamespace(
        rollout=SimpleNamespace(
            chunk_size_frames=2,
            chunk_overlap_frames=0,
            decode_mode="causal_chunk",
            reward_eps=1e-8,
        ),
        plasticity=SimpleNamespace(max_fast_rank=3, max_fast_norm_ratio=10.0),
        eggroll=SimpleNamespace(sigma=0.05),
    )


def _decode_from_logits(output, tokenizer, batch_size, num_candidates):
    del tokenizer
    logits = output["final_posteriors"].reshape(batch_size, num_candidates, -1, 2)
    vals = logits.sum(dim=(-1, -2))
    return [[f"{vals[b, n].item():.6f}" for n in range(num_candidates)] for b in range(batch_size)]


def _wer_from_numeric_chunks(hyp, ref):
    del ref
    if not hyp:
        return 0.0
    return abs(sum(float(part) for part in hyp.split())) * 0.01


def test_batched_rollout_matches_serial_candidate_reference(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(gpu_plasticity, "decode_output", _decode_from_logits)
    base_asr = TinyASR()
    module_specs = wrap_linear_modules(base_asr, ["adapt"])
    updater = PlasticityPolicy(
        module_specs,
        token_dim=4,
        comm_dim=4,
        update_rank=1,
        max_eta=1e-3,
        default_rho=0.8,
    )
    perturbations = sample_rank1_perturbations(updater, 4, antithetic=True)
    audio = torch.randn(2, 2, 4)
    refs = ["", ""]

    batched_rewards, batched_info = rollout_recordings_with_plasticity_candidates(
        asr_model=copy.deepcopy(base_asr),
        updater=copy.deepcopy(updater),
        recording_audio_batch=audio,
        reference_text_batch=refs,
        tokenizer=None,
        candidate_perturbations=perturbations,
        config=_config(),
        module_specs=module_specs,
        wer_fn=_wer_from_numeric_chunks,
    )

    serial_wers = []
    for nidx in range(perturbations.num_candidates):
        one_candidate = EggrollPerturbations(
            {
                name: Rank1Perturbation(
                    a=pert.a[nidx : nidx + 1],
                    b=pert.b[nidx : nidx + 1],
                    antithetic=False,
                )
                for name, pert in perturbations.items()
            }
        )
        _, info = rollout_recordings_with_plasticity_candidates(
            asr_model=copy.deepcopy(base_asr),
            updater=copy.deepcopy(updater),
            recording_audio_batch=audio,
            reference_text_batch=refs,
            tokenizer=None,
            candidate_perturbations=one_candidate,
            config=_config(),
            module_specs=module_specs,
            wer_fn=_wer_from_numeric_chunks,
        )
        serial_wers.append(info["wer"][:, 0])
    serial_wers = torch.stack(serial_wers, dim=1)
    serial_quality = 1.0 - serial_wers.clamp(max=1.0)
    serial_rewards = group_normalise_rewards(serial_quality).mean(dim=0)

    torch.testing.assert_close(batched_info["wer"], serial_wers, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_rewards, serial_rewards, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_info["chunks_per_recording"], torch.tensor([2, 2]))
    assert batched_info["rollout_chunk_steps"].item() == 2
    assert batched_info["rollout_streams"].item() == 8
    assert batched_info["chunk_size_frames"].item() == 2


def test_inner_forward_functions_do_not_accept_label_or_reward_inputs():
    forbidden = {"reference_text", "references", "wer", "cer", "pseudo_labels", "reward", "rewards"}
    for fn in [
        PlasticityPolicy.forward,
        plasticity.asr_forward_with_fast_state,
        plasticity.apply_fast_updates,
    ]:
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden)
