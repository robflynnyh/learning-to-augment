from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from l2augment.modelling.plasticity import (
    AdaptedLinearSpec,
    FastWeightState,
    PlasticityPolicy,
    apply_fast_updates,
    asr_forward_with_fast_state,
    init_fast_state,
)
from l2augment.utils.eggroll import EggrollPerturbations, group_normalise_rewards


def _cfg_get(config, path: str, default=None):
    cur = config
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def segment_recording(audio: Tensor, chunk_size: int, overlap: int = 0) -> Tuple[Tensor, Tensor]:
    if audio.dim() != 2:
        raise ValueError("audio must have shape [C, S]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    C, S = audio.shape
    stride = chunk_size - overlap
    chunks: List[Tensor] = []
    lengths: List[int] = []
    for start in range(0, S, stride):
        chunk = audio[:, start : start + chunk_size]
        length = chunk.shape[-1]
        if length == 0:
            continue
        if length < chunk_size:
            chunk = torch.cat(
                [chunk, torch.zeros(C, chunk_size - length, device=audio.device, dtype=audio.dtype)],
                dim=-1,
            )
        chunks.append(chunk)
        lengths.append(length)
        if start + chunk_size >= S:
            break
    return torch.stack(chunks, dim=0), torch.tensor(lengths, device=audio.device)


def segment_batch(
    recording_audio_batch: Tensor | Sequence[Tensor],
    *,
    chunk_size: int,
    overlap: int = 0,
) -> Tuple[Tensor, Tensor]:
    if isinstance(recording_audio_batch, Tensor):
        recordings = [recording_audio_batch[i] for i in range(recording_audio_batch.shape[0])]
    else:
        recordings = list(recording_audio_batch)
    segmented = [segment_recording(audio, chunk_size, overlap) for audio in recordings]
    max_chunks = max(chunks.shape[0] for chunks, _ in segmented)
    C = segmented[0][0].shape[1]
    chunk_size = segmented[0][0].shape[2]
    device = segmented[0][0].device
    dtype = segmented[0][0].dtype
    chunks_out = torch.zeros(len(segmented), max_chunks, C, chunk_size, device=device, dtype=dtype)
    lengths_out = torch.zeros(len(segmented), max_chunks, device=device, dtype=torch.long)
    for bidx, (chunks, lengths) in enumerate(segmented):
        chunks_out[bidx, : chunks.shape[0]] = chunks
        lengths_out[bidx, : lengths.shape[0]] = lengths
    return chunks_out, lengths_out


def decode_output(output, tokenizer, batch_size: int, num_candidates: int) -> List[List[str]]:
    try:
        from lcasr.decoding.greedy import GreedyCTCDecoder
    except Exception as exc:  # pragma: no cover - only exercised without lcasr installed
        raise RuntimeError("lcasr is required for tokenizer-based ASR decoding") from exc

    posteriors = output["final_posteriors"]
    if posteriors.shape[0] != batch_size * num_candidates:
        raise ValueError("ASR output batch does not match B*N")
    blank_id = posteriors.shape[-1] - 1
    if hasattr(tokenizer, "vocab_size"):
        blank_id = tokenizer.vocab_size()
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=blank_id)
    decoded = [decoder(posteriors[i]) for i in range(posteriors.shape[0])]
    return [
        decoded[bidx * num_candidates : (bidx + 1) * num_candidates]
        for bidx in range(batch_size)
    ]


def compute_wer_matrix(
    transcript_parts: Sequence[Sequence[Sequence[str]]],
    reference_text_batch: Sequence[str],
    *,
    wer_fn: Optional[Callable[[str, str], float]] = None,
) -> Tensor:
    if wer_fn is None:
        try:
            from lcasr.eval.wer import word_error_rate_detail
        except Exception as exc:  # pragma: no cover - only exercised without lcasr installed
            raise RuntimeError("lcasr is required for WER computation") from exc

        def wer_fn(hyp: str, ref: str) -> float:
            return float(word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)[0])

    B = len(transcript_parts)
    N = len(transcript_parts[0]) if B else 0
    wers = torch.zeros(B, N)
    for bidx in range(B):
        for nidx in range(N):
            hyp = " ".join(part for part in transcript_parts[bidx][nidx] if part)
            wers[bidx, nidx] = float(wer_fn(hyp, reference_text_batch[bidx]))
    return wers


def rollout_recordings_with_plasticity_candidates(
    *,
    asr_model: nn.Module,
    updater: PlasticityPolicy,
    recording_audio_batch: Tensor | Sequence[Tensor],
    reference_text_batch: Sequence[str],
    tokenizer,
    candidate_perturbations: EggrollPerturbations,
    config,
    module_specs: Mapping[str, Tuple[int, int] | AdaptedLinearSpec],
    wer_fn: Optional[Callable[[str, str], float]] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    decode_mode = _cfg_get(config, "rollout.decode_mode", "causal_chunk")
    if decode_mode != "causal_chunk":
        raise ValueError("MVP plasticity rollout only supports causal_chunk decode_mode")

    chunk_size = _cfg_get(config, "rollout.chunk_size_frames", None)
    if chunk_size is None:
        chunk_size = _cfg_get(config, "rollout.chunk_size_seconds", None)
    if chunk_size is None:
        raise ValueError("config.rollout.chunk_size_frames is required")
    overlap = _cfg_get(config, "rollout.chunk_overlap_frames", 0)
    if overlap == 0:
        overlap = _cfg_get(config, "rollout.chunk_overlap_seconds", 0)
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    pass_lengths = bool(_cfg_get(config, "rollout.pass_lengths", False))

    chunks, chunk_lengths = segment_batch(
        recording_audio_batch,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    B, T, C, S = chunks.shape
    N = candidate_perturbations.num_candidates
    device = chunks.device
    dtype = chunks.dtype
    sigma = float(_cfg_get(config, "eggroll.sigma", 0.0))

    fast_state = init_fast_state(
        batch_size=B,
        num_candidates=N,
        module_specs=module_specs,
        device=device,
        dtype=dtype,
    )
    transcript_parts: List[List[List[str]]] = [[[] for _ in range(N)] for _ in range(B)]

    asr_model.eval()
    updater.eval()
    with torch.no_grad():
        for t in range(T):
            chunk_t = chunks[:, t]
            length_t = chunk_lengths[:, t]
            chunk_bn = chunk_t[:, None].expand(B, N, C, S).contiguous()
            length_bn = length_t[:, None].expand(B, N).contiguous()

            output, activations = asr_forward_with_fast_state(
                asr_model=asr_model,
                audio=chunk_bn,
                lengths=length_bn if pass_lengths else None,
                fast_state=fast_state,
                batch_size=B,
                num_candidates=N,
                return_selected_activations=True,
            )
            decoded = decode_output(output, tokenizer, B, N)
            for bidx in range(B):
                for nidx in range(N):
                    transcript_parts[bidx][nidx].append(decoded[bidx][nidx])

            updates = updater(
                activations=activations,
                fast_state=fast_state,
                perturbations=candidate_perturbations,
                sigma=sigma,
                config=config,
            )
            fast_state = apply_fast_updates(
                fast_state=fast_state,
                updates=updates,
                max_fast_rank=int(_cfg_get(config, "plasticity.max_fast_rank", 8)),
                max_fast_norm_ratio=float(_cfg_get(config, "plasticity.max_fast_norm_ratio", 1e-3)),
            )

    wers = compute_wer_matrix(transcript_parts, reference_text_batch, wer_fn=wer_fn).to(device)
    quality = 1.0 - wers.clamp(max=1.0)
    rewards_bn = group_normalise_rewards(quality, eps=float(_cfg_get(config, "rollout.reward_eps", 1e-8)))
    reward_per_candidate = rewards_bn.mean(dim=0)
    return reward_per_candidate, {
        "wer": wers,
        "quality": quality,
        "rewards_bn": rewards_bn,
        "final_fast_state_rank": torch.tensor(
            [fast_state[name].A.shape[-1] for name in fast_state.keys()],
            device=device,
        ),
    }


def serial_reference_rollout(
    *,
    asr_model_factory: Callable[[], nn.Module],
    updater_factory: Callable[[], PlasticityPolicy],
    recording_audio_batch: Tensor,
    reference_text_batch: Sequence[str],
    tokenizer,
    perturbations: EggrollPerturbations,
    config,
    module_specs: Mapping[str, Tuple[int, int] | AdaptedLinearSpec],
    wer_fn: Callable[[str, str], float],
):
    rewards = []
    info_by_candidate = []
    for nidx in range(perturbations.num_candidates):
        one_layers = {
            name: type(pert)(
                a=pert.a[nidx : nidx + 1],
                b=pert.b[nidx : nidx + 1],
                antithetic=False,
            )
            for name, pert in perturbations.items()
        }
        reward, info = rollout_recordings_with_plasticity_candidates(
            asr_model=asr_model_factory(),
            updater=updater_factory(),
            recording_audio_batch=recording_audio_batch,
            reference_text_batch=reference_text_batch,
            tokenizer=tokenizer,
            candidate_perturbations=EggrollPerturbations(one_layers),
            config=config,
            module_specs=module_specs,
            wer_fn=wer_fn,
        )
        rewards.append(reward[0])
        info_by_candidate.append(info)
    return torch.stack(rewards), info_by_candidate
