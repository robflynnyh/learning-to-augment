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




    asr_model.eval()

    if audio_a is None or audio_b is None:
        print('--')
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

    prev_cer = word_error_rate_detail(hypotheses=[normalize(out_original_predictions)], references=[normalize(teacher_targets)], use_cer=True)[0] * 100
    updated_cer = word_error_rate_detail(hypotheses=[normalize(updated_predictions)], references=[normalize(teacher_targets)], use_cer=True)[0] * 100
    # prev_ctc_loss = ctc_loss_fn(out_original_posteriors, teacher_logits, torch.LongTensor([N] * 1), torch.LongTensor([teacher_logits.shape[0]] * 1)) / total_tokens_in_loss
    # updated_ctc_loss = ctc_loss_fn(updated_logits, teacher_logits, torch.LongTensor([N] * 1), torch.LongTensor([teacher_logits.shape[0]] * 1)) / total_tokens_in_loss

    diff = prev_cer - updated_cer
    print(diff)
    
    return torch.tensor(prev_cer), torch.tensor(updated_cer), masks
        
       









