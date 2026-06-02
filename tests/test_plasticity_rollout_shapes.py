import copy
import inspect
from types import SimpleNamespace

import torch
from torch import nn

from l2augment.modelling import plasticity
from l2augment.modelling.plasticity import (
    PlasticityPolicy,
    apply_fast_updates,
    asr_forward_with_fast_state,
    discover_attention_out_proj_modules,
    init_fast_state,
    wrap_linear_modules,
)
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
        self.batch_sizes = []

    def forward(self, audio_signal, length=None):
        del length
        self.batch_sizes.append(audio_signal.shape[0])
        x = audio_signal.transpose(1, 2)
        y = self.adapt(x)
        return {"final_posteriors": y}


class TinyAttendFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(2, 2)

    def forward(self, x):
        return self.out_proj(x)


class TinyAttend(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = TinyAttendFn()

    def forward(self, x):
        return self.fn(x)


class TinyEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attend = TinyAttend()

    def forward(self, x):
        return self.attend(x)


class TinyLayeredASR(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([TinyEncoderLayer() for _ in range(num_layers)])

    def forward(self, audio_signal, length=None):
        del length
        x = audio_signal.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return {"final_posteriors": x}


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


def _decode_one_token_per_stream(output, tokenizer, batch_size, num_candidates):
    del output, tokenizer
    return [["1" for _ in range(num_candidates)] for _ in range(batch_size)]


def _wer_token_count(hyp, ref):
    del ref
    return float(len(hyp.split()))


class _StrictTokenizer:
    def vocab_size(self):
        return 99

    def decode(self, token_ids):
        if any(token_id >= 2 for token_id in token_ids):
            raise AssertionError(f"decode received non-token ids: {token_ids}")
        return "".join(str(token_id) for token_id in token_ids)


def test_decode_output_uses_model_output_blank_and_cpu_ctc_collapse():
    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
                [0.0, 0.0, 5.0],
                [4.0, 0.0, 0.0],
            ]
        ]
    )

    decoded = gpu_plasticity.decode_output(
        {"final_posteriors": logits},
        _StrictTokenizer(),
        batch_size=1,
        num_candidates=1,
    )

    assert decoded == [["10"]]


def test_segment_batch_respects_true_lengths_for_padded_tensor_batch():
    audio = torch.zeros(2, 1, 6)
    audio[0, :, :3] = 1.0
    audio[1, :, :5] = 2.0

    _, lengths = gpu_plasticity.segment_batch(
        audio,
        chunk_size=2,
        overlap=0,
        recording_lengths=torch.tensor([3, 5]),
    )

    torch.testing.assert_close(lengths, torch.tensor([[2, 1, 0], [2, 2, 1]]))


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


def test_rollout_does_not_decode_or_forward_padded_recording_chunks(monkeypatch):
    torch.manual_seed(1)
    monkeypatch.setattr(gpu_plasticity, "decode_output", _decode_one_token_per_stream)
    asr = TinyASR()
    module_specs = wrap_linear_modules(asr, ["adapt"])
    updater = PlasticityPolicy(
        module_specs,
        token_dim=4,
        comm_dim=4,
        update_rank=1,
        max_eta=1e-3,
        default_rho=0.8,
    )
    perturbations = sample_rank1_perturbations(updater, 2, antithetic=True)
    audio = torch.randn(2, 2, 4)

    _, info = rollout_recordings_with_plasticity_candidates(
        asr_model=asr,
        updater=updater,
        recording_audio_batch=audio,
        recording_lengths=torch.tensor([2, 4]),
        reference_text_batch=["", ""],
        tokenizer=None,
        candidate_perturbations=perturbations,
        config=_config(),
        module_specs=module_specs,
        wer_fn=_wer_token_count,
    )

    torch.testing.assert_close(info["chunks_per_recording"], torch.tensor([1, 2]))
    torch.testing.assert_close(info["wer"], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert asr.batch_sizes == [4, 2]


def test_multi_target_attention_out_proj_path_captures_and_updates_all_modules():
    torch.manual_seed(2)
    model = TinyLayeredASR(num_layers=3)
    target_modules = discover_attention_out_proj_modules(model)

    assert target_modules == [
        "layers.0.attend.fn.out_proj",
        "layers.1.attend.fn.out_proj",
        "layers.2.attend.fn.out_proj",
    ]

    module_specs = wrap_linear_modules(model, target_modules)
    assert list(module_specs) == target_modules

    updater = PlasticityPolicy(
        module_specs,
        token_dim=4,
        comm_dim=4,
        update_rank=1,
        max_eta=1e-3,
        default_rho=0.8,
    )
    fast_state = init_fast_state(
        batch_size=2,
        num_candidates=2,
        module_specs=module_specs,
        device="cpu",
    )
    audio = torch.randn(2, 2, 2, 5)

    _, activations = asr_forward_with_fast_state(
        asr_model=model,
        audio=audio,
        lengths=None,
        fast_state=fast_state,
        batch_size=2,
        num_candidates=2,
        return_selected_activations=True,
    )
    assert list(activations) == target_modules

    layer_tokens = updater.project_layer_tokens(activations)
    assert layer_tokens.shape == (2, 2, 3, 4)

    updates = updater(activations, fast_state=fast_state)
    assert list(updates) == target_modules

    next_state = apply_fast_updates(
        fast_state,
        updates,
        max_fast_rank=4,
        max_fast_norm_ratio=10.0,
    )
    assert {name: next_state[name].A.shape[-1] for name in target_modules} == {
        name: 1 for name in target_modules
    }


def test_inner_forward_functions_do_not_accept_label_or_reward_inputs():
    forbidden = {"reference_text", "references", "wer", "cer", "pseudo_labels", "reward", "rewards"}
    for fn in [
        PlasticityPolicy.forward,
        plasticity.asr_forward_with_fast_state,
        plasticity.apply_fast_updates,
    ]:
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden)
