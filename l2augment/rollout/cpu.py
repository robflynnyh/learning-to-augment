import torch
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail

from l2augment.utils.data import prepare_chunks

import random
from functools import partial
from einops import repeat, rearrange
import math
from tqdm import tqdm

DEFAULT_OPTIMIZER_CLASS = torch.optim.SGD

def apply_ctc_loss_fn(ctc_loss_fn, pseudo_targets, a_lengths, target_lengths, total_tokens_in_loss, noisy_predictions):
  return ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, a_lengths, target_lengths) / total_tokens_in_loss


def cpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        audio:Tensor,
        teacher_logits:Tensor,
        seq_len:int=4096,
        overlap=0.875,
        optim_args:Dict[str, Any] = {"lr":1e-3},
        max_steps = None,
        **kwargs
    ):

    dtype = torch.float32 #temporary

    overlap = round(seq_len*overlap)
    audio = audio.to(dtype=dtype) #temporary
    audio_n = audio.shape[-1]

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    
    asr_model = load_asr_model_fn()
    asr_model = asr_model.to(dtype=dtype)
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')


    if seq_len > audio_n:
        seq_len, overlap = audio_n, 0

    all_logits, logit_count = torch.zeros((1, audio_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, audio_n//4 + seq_len, tokenizer.vocab_size() + 1))

    training_data, training_keys = prepare_chunks(audio, seq_len, overlap)
    training_keys = list(training_data.keys())

    model_outputs = {}
    asr_model.eval()
    

    policy_states = None
    policy_cache = None
    masks = []
    logit_position = 0
    rewards = []
    #lr_indexes = []
    total_steps = len(training_keys)
    if max_steps != None and max_steps < len(training_keys): total_steps = max_steps
    for i, key in tqdm(enumerate(training_keys), total=total_steps):
        if max_steps != None and i > max_steps: break

        audio_chunk = training_data[key].clone()
        with torch.no_grad(): policy_output = policy.augment(audio_chunk, state=policy_states, cache=policy_cache)
    
        audio_teacher, audio_student = policy_output['augmented_data'].chunk(2, dim=0)

        policy_states = policy_output['next_state']
        policy_cache = policy_output['next_cache']

        masks.append(policy_output['masks'].to(dtype=torch.bool))

        with torch.no_grad():
            out_original = asr_model(audio_signal = audio_chunk)
            out_teacher = asr_model(audio_signal = audio_teacher)
        out_noised = asr_model(audio_signal = audio_student)['final_posteriors']

        out_original_posteriors = out_original['final_posteriors'].squeeze(0)
        teacher_output_posteriors = out_teacher['final_posteriors'].squeeze(0)
        pseudo_targets = decoder(teacher_output_posteriors) 
        pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets))[None]
        N, B = out_noised.shape[1], out_noised.shape[0]
        total_tokens_in_loss = N * B
        loss = ctc_loss_fn(out_noised.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * out_noised.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss
        optimizer.zero_grad()
        loss.backward()

        if kwargs.get("clip", False): torch.nn.utils.clip_grad_norm_(asr_model.parameters(), kwargs["clip"]) 

        optimizer.step()

        with torch.no_grad():
            out_updated = asr_model(audio_signal = audio_chunk)
        updated_logits = out_updated['final_posteriors'].squeeze(0) # BNC -> NC

        ds_len = updated_logits.shape[-2]
        teacher_logits_at_t = teacher_logits[0, logit_position:logit_position+ds_len]
        #print(teacher_output_posteriors.shape, teacher_logits_at_t.shape, teacher_logits.shape)
        
        prev_kl_div = torch.nn.functional.kl_div(out_original_posteriors, teacher_logits_at_t, reduction='none', log_target=True)
        updated_kl_div = torch.nn.functional.kl_div(updated_logits, teacher_logits_at_t, reduction='none', log_target=True)
        kl_diff = (prev_kl_div - updated_kl_div)
    
        rewards.append(kl_diff)
        
        #print(kl_diff.mean())
        ratio = audio_chunk.shape[-1] / ds_len
        overlap_ds = int(overlap / ratio)
        #print(logit_position, ds_len, overlap_ds)
        logit_position += ds_len - overlap_ds

    masks = torch.cat(masks, 0).squeeze(-1)
    rewards = torch.stack(rewards)
    print(masks.shape, rewards.shape)
    #lrs = torch.stack(lr_indexes, 0).squeeze(-1)

    return {
        'masks': masks,
        'rewards': rewards,
        #'lrs':lrs,
    }

 
        
       









