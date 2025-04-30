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
from whisper.normalizers import EnglishTextNormalizer

normalize = EnglishTextNormalizer()

DEFAULT_OPTIMIZER_CLASS = torch.optim.SGD

def apply_ctc_loss_fn(ctc_loss_fn, pseudo_targets, a_lengths, target_lengths, total_tokens_in_loss, noisy_predictions):
  return ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, a_lengths, target_lengths) / total_tokens_in_loss


def cpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        audio:Tensor,
        text:str,
        optim_args:Dict[str, Any] = {"lr":1e-1},
        audio_a:Tensor = None,
        audio_b:Tensor = None,
        augmented_target = False,
        return_wer = False,
        verbose=True,
        get_weight_difference = False,
        return_masked_prediction = False,
        **kwargs
    ):

    dtype = torch.float32 #temporary
    device = audio.device

    audio = audio.to(dtype=dtype) #temporary

    #torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    
    asr_model = load_asr_model_fn()
    asr_model = asr_model.to(dtype=dtype).to(device)
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')

    if get_weight_difference:
        # Get the weights of the model before training
        initial_weights = {name: param.clone() for name, param in asr_model.named_parameters()}


    asr_model.eval()
    

    if audio_a is None or audio_b is None:
        print('--') if verbose else None
        with torch.no_grad(): policy_output = policy.augment(audio)
        audio_a, audio_b = policy_output['augmented_data'].chunk(2, dim=0)
        masks = policy_output['masks']
    else: masks = None

  
    with torch.no_grad():
        out_original = asr_model(audio_signal = audio)
        if augmented_target: out_a = asr_model(audio_signal = audio_a)['final_posteriors']
        else: out_a = out_original['final_posteriors'].clone()
    
    out_b = asr_model(audio_signal = audio_b)['final_posteriors']

    out_original_posteriors = out_original['final_posteriors'].squeeze(0)
    out_original_predictions = decoder(out_original_posteriors)
    

    pseudo_targets = decoder(out_a.squeeze(0)) 
    pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets))[None]




    N, B = out_a.shape[1], out_a.shape[0]
    total_tokens_in_loss = N * B
    #loss_a = torch.nn.functional.kl_div(out_a, out_b, reduction='sum', log_target=True) / total_tokens_in_loss

    loss_b = ctc_loss_fn(out_b.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * out_b.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss
    loss = loss_b#(loss_a + loss_b)/2

    optimizer.zero_grad()
    loss.backward()

    if kwargs.get("clip", False): torch.nn.utils.clip_grad_norm_(asr_model.parameters(), kwargs["clip"]) 

    optimizer.step()

    with torch.no_grad():
        out_updated = asr_model(audio_signal = audio)
    updated_logits = out_updated['final_posteriors'].squeeze(0) # BNC -> NC
    updated_predictions = decoder(updated_logits)

    teacher_targets = text
   
    
    #print(teacher_output_posteriors.shape, teacher_logits_at_t.shape, teacher_logits.shape)

    prev = word_error_rate_detail(hypotheses=[normalize(out_original_predictions)], references=[normalize(teacher_targets)], use_cer=True)[0] * 100
    updated = word_error_rate_detail(hypotheses=[normalize(updated_predictions)], references=[normalize(teacher_targets)], use_cer=True)[0] * 100
    # prev_ctc_loss = ctc_loss_fn(out_original_posteriors, teacher_logits, torch.LongTensor([N] * 1), torch.LongTensor([teacher_logits.shape[0]] * 1)) / total_tokens_in_loss
    # updated_ctc_loss = ctc_loss_fn(updated_logits, teacher_logits, torch.LongTensor([N] * 1), torch.LongTensor([teacher_logits.shape[0]] * 1)) / total_tokens_in_loss

    diff = prev - updated
    print(diff) if verbose else None

    prev, updated = torch.tensor(prev), torch.tensor(updated)

    if return_wer:
        prev_wer = word_error_rate_detail(hypotheses=[normalize(out_original_predictions)], references=[normalize(teacher_targets)], use_cer=False)[0] * 100
        updated_wer = word_error_rate_detail(hypotheses=[normalize(updated_predictions)], references=[normalize(teacher_targets)], use_cer=False)[0] * 100
        diff = prev_wer - updated_wer
        prev_wer, updated_wer = torch.tensor(prev_wer), torch.tensor(updated_wer)

        prev = torch.stack([prev, prev_wer])
        updated = torch.stack([updated, updated_wer])
        print(prev.shape, updated.shape) if verbose else None

    returns = [prev, updated, masks]

    if get_weight_difference:
        # Get the weights of the model after training
        final_weights = {name: param.clone() for name, param in asr_model.named_parameters()}
        weight_diff = {name: (final_weights[name] - initial_weights[name]).norm().item() for name in initial_weights.keys()}
        total_weight_diff = sum(weight_diff.values())
        print(f"Total weight difference: {total_weight_diff}") if verbose else None
        returns.append(total_weight_diff)
    
    if return_masked_prediction   :
        returns.append(out_b)

    return returns
        
       









