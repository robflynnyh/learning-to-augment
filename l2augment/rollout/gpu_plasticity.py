from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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


def segment_recording(
    audio: Tensor,
    chunk_size: int,
    overlap: int = 0,
    *,
    recording_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    if audio.dim() != 2:
        raise ValueError("audio must have shape [C, S]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    C, S = audio.shape
    if recording_length is not None:
        S = int(recording_length)
        if S <= 0:
            raise ValueError("recording_length must be positive")
        if S > audio.shape[-1]:
            raise ValueError("recording_length cannot exceed audio length")
        audio = audio[:, :S]
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
    recording_lengths: Optional[Tensor | Sequence[int]] = None,
) -> Tuple[Tensor, Tensor]:
    if isinstance(recording_audio_batch, Tensor):
        recordings = [recording_audio_batch[i] for i in range(recording_audio_batch.shape[0])]
    else:
        recordings = list(recording_audio_batch)
    if recording_lengths is None:
        lengths = [None] * len(recordings)
    else:
        lengths = [int(length) for length in recording_lengths]
        if len(lengths) != len(recordings):
            raise ValueError("recording_lengths must match the recording batch size")
    segmented = [
        segment_recording(audio, chunk_size, overlap, recording_length=length)
        for audio, length in zip(recordings, lengths)
    ]
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
    posteriors = output["final_posteriors"]
    if posteriors.shape[0] != batch_size * num_candidates:
        raise ValueError("ASR output batch does not match B*N")
    blank_id = posteriors.shape[-1] - 1
    predictions = posteriors.detach().argmax(dim=-1).to("cpu")
    decoded = []
    for indices in predictions:
        collapsed = torch.unique_consecutive(indices, dim=-1).tolist()
        token_ids = [idx for idx in collapsed if idx != blank_id]
        decoded.append(tokenizer.decode(token_ids) if tokenizer is not None else token_ids)
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


def _select_fast_state_recordings(fast_state: FastWeightState, indexes: Tensor) -> FastWeightState:
    return FastWeightState(
        {
            name: type(factors)(
                A=factors.A.index_select(0, indexes),
                B=factors.B.index_select(0, indexes),
                base_weight_norm=factors.base_weight_norm,
            )
            for name, factors in fast_state.items()
        }
    )


def _scatter_fast_state_recordings(
    fast_state: FastWeightState,
    updated_subset: FastWeightState,
    indexes: Tensor,
) -> FastWeightState:
    next_factors = {}
    for name, factors in fast_state.items():
        updated = updated_subset[name]
        target_rank = max(factors.A.shape[-1], updated.A.shape[-1])
        if target_rank == factors.A.shape[-1]:
            A = factors.A.clone()
            Bf = factors.B.clone()
        else:
            A = factors.A.new_zeros(*factors.A.shape[:-1], target_rank)
            Bf = factors.B.new_zeros(*factors.B.shape[:-1], target_rank)
            A[..., : factors.A.shape[-1]] = factors.A
            Bf[..., : factors.B.shape[-1]] = factors.B
        if updated.A.shape[-1] != target_rank:
            A_update = updated.A.new_zeros(*updated.A.shape[:-1], target_rank)
            B_update = updated.B.new_zeros(*updated.B.shape[:-1], target_rank)
            A_update[..., : updated.A.shape[-1]] = updated.A
            B_update[..., : updated.B.shape[-1]] = updated.B
        else:
            A_update = updated.A
            B_update = updated.B
        A.index_copy_(0, indexes, A_update)
        Bf.index_copy_(0, indexes, B_update)
        next_factors[name] = type(factors)(
            A=A,
            B=Bf,
            base_weight_norm=factors.base_weight_norm,
        )
    return FastWeightState(next_factors)


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
    recording_lengths: Optional[Tensor | Sequence[int]] = None,
    wer_fn: Optional[Callable[[str, str], float]] = None,
    progress_log_fn: Optional[Callable[[Mapping[str, Any]], None]] = None,
    progress_context: Optional[Mapping[str, Any]] = None,
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
    progress_every = int(_cfg_get(config, "rollout.progress_log_every_chunks", 0) or 0)

    chunks, chunk_lengths = segment_batch(
        recording_audio_batch,
        chunk_size=chunk_size,
        overlap=overlap,
        recording_lengths=recording_lengths,
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
    chunks_per_recording = (chunk_lengths > 0).sum(dim=1)
    progress_base = dict(progress_context or {})
    progress_base.update(
        {
            "B": B,
            "N": N,
            "T": T,
            "device": str(device),
            "rollout_streams": B * N,
            "chunks_per_recording_min": int(chunks_per_recording.min().detach().cpu()),
            "chunks_per_recording_max": int(chunks_per_recording.max().detach().cpu()),
            "chunks_per_recording_mean": float(chunks_per_recording.detach().float().mean().cpu()),
        }
    )

    def log_progress(event: str, **fields: Any) -> None:
        if progress_log_fn is None:
            return
        payload = dict(progress_base)
        payload.update({"event": event, **fields})
        progress_log_fn(payload)

    rollout_start = time.monotonic()
    log_progress("plasticity_rollout_start")

    asr_model.eval()
    updater.eval()
    with torch.no_grad():
        for t in range(T):
            chunk_t = chunks[:, t]
            length_t = chunk_lengths[:, t]
            active_indexes = (length_t > 0).nonzero(as_tuple=False).flatten()
            if active_indexes.numel() == 0:
                continue

            B_active = int(active_indexes.numel())
            chunk_active = chunk_t.index_select(0, active_indexes)
            length_active = length_t.index_select(0, active_indexes)
            chunk_bn = chunk_active[:, None].expand(B_active, N, C, S).contiguous()
            length_bn = length_active[:, None].expand(B_active, N).contiguous()
            active_fast_state = _select_fast_state_recordings(fast_state, active_indexes)

            output, activations = asr_forward_with_fast_state(
                asr_model=asr_model,
                audio=chunk_bn,
                lengths=length_bn if pass_lengths else None,
                fast_state=active_fast_state,
                batch_size=B_active,
                num_candidates=N,
                return_selected_activations=True,
            )
            decoded = decode_output(output, tokenizer, B_active, N)
            for active_bidx, original_bidx in enumerate(active_indexes.tolist()):
                for nidx in range(N):
                    transcript_parts[original_bidx][nidx].append(decoded[active_bidx][nidx])

            updates = updater(
                activations=activations,
                fast_state=active_fast_state,
                perturbations=candidate_perturbations,
                sigma=sigma,
                config=config,
            )
            updated_active_fast_state = apply_fast_updates(
                fast_state=active_fast_state,
                updates=updates,
                max_fast_rank=int(_cfg_get(config, "plasticity.max_fast_rank", 8)),
                max_fast_norm_ratio=float(_cfg_get(config, "plasticity.max_fast_norm_ratio", 1e-3)),
            )
            fast_state = _scatter_fast_state_recordings(
                fast_state,
                updated_active_fast_state,
                active_indexes,
            )
            if progress_every > 0 and (
                t == 0 or (t + 1) % progress_every == 0 or t + 1 == T
            ):
                log_progress(
                    "plasticity_rollout_chunk",
                    chunk_index=t + 1,
                    elapsed_seconds=round(time.monotonic() - rollout_start, 3),
                )

    wers = compute_wer_matrix(transcript_parts, reference_text_batch, wer_fn=wer_fn).to(device)
    quality = 1.0 - wers.clamp(max=1.0)
    rewards_bn = group_normalise_rewards(quality, eps=float(_cfg_get(config, "rollout.reward_eps", 1e-8)))
    reward_per_candidate = rewards_bn.mean(dim=0)
    valid_chunk_lengths = chunk_lengths[chunk_lengths > 0]
    if valid_chunk_lengths.numel() == 0:
        valid_chunk_lengths = torch.zeros(1, device=device, dtype=chunk_lengths.dtype)
    return reward_per_candidate, {
        "wer": wers,
        "quality": quality,
        "rewards_bn": rewards_bn,
        "chunks_per_recording": chunks_per_recording.to(device),
        "chunk_length_frames_mean": valid_chunk_lengths.float().mean().to(device),
        "chunk_size_frames": torch.tensor(chunk_size, device=device),
        "chunk_overlap_frames": torch.tensor(overlap, device=device),
        "rollout_chunk_steps": torch.tensor(T, device=device),
        "rollout_streams": torch.tensor(B * N, device=device),
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
