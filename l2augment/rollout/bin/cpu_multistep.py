import torch
from madgrad import MADGRAD
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

import random
from functools import partial
from einops import repeat, rearrange
import math
from tqdm import tqdm

DEFAULT_OPTIMIZER_CLASS = MADGRAD

def apply_ctc_loss_fn(ctc_loss_fn, pseudo_targets, a_lengths, target_lengths, total_tokens_in_loss, noisy_predictions):
  return ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, a_lengths, target_lengths) / total_tokens_in_loss


def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())

def broadcast_multiply(tensor1, tensor2):
    """
    Multiply a tensor of size B with a tensor of size (B, ...) 
    where the remaining dimensions can vary.
    
    Args:
    - tensor1 (torch.Tensor): Tensor of size B
    - tensor2 (torch.Tensor): Tensor of size (B, ...)
    
    Returns:
    torch.Tensor: Result of broadcasting multiplication
    """
    # Reshape tensor1 to have additional singleton dimensions for broadcasting
    # This ensures tensor1 can be multiplied with tensor2
    broadcast_tensor1 = tensor1.view(tensor1.size(0), *([1] * (tensor2.ndim - 1)))
    # Multiply with broadcasting
    return broadcast_tensor1 * tensor2

def cpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        audio:Tensor,
        text:str,
        seq_len:int=4096,
        overlap=0.875,
        optim_args:Dict[str, Any] = {"lr":1e-5},
        original_wer = None,
        max_steps = None,
        **kwargs
    ):

    dtype = torch.float32 #temporary

    overlap = round(seq_len*overlap)
    audio = audio.to(dtype=dtype) #temporary
    audio_n = audio.shape[-1]
    
    text = normalizer(text)
    
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
    
    if original_wer is None:
        for key in tqdm(training_keys):
            audio_chunk = training_data[key].clone()
            with torch.no_grad():
                out = asr_model(audio_signal = audio_chunk)
            logits = out['final_posteriors'][0]
            logits = torch.exp(logits) # convert to prob
            ds_len = logits.shape[-2]
            ratio = audio_chunk.shape[-1] / ds_len
            overlap_ds = int(overlap / ratio)
            model_outputs[key] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

        logit_position = 0
        for i in sorted(list(model_outputs.keys())):
            logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
            logit_position -= overlap_ds if i != 0 else 0
            logit_count[:, logit_position:logit_position+ds_len, :] += 1
            all_logits[:, logit_position:logit_position+ds_len, :] += logits
            logit_position += ds_len     

        B,N,C = all_logits.shape
        all_logits = all_logits[logit_count.sum(dim=-1) != 0]
        all_logits = all_logits.reshape(B,-1,C)
        logit_count = logit_count[logit_count.sum(dim=-1) != 0]
        logit_count = logit_count.reshape(B,-1,C)
        logits = all_logits / logit_count
        logits = torch.log(logits) 
        out_text = decoder(logits.squeeze())
        out_text = normalizer(out_text)
        original_wer = word_error_rate_detail(hypotheses=[out_text], references=[text], use_cer=True)[0]
    

    policy_states = None
    seeds = []
    masks = []
    lr_indexes = []
    total_steps = len(training_keys)
    if max_steps != None and max_steps < len(training_keys): total_steps = max_steps
    for i, key in tqdm(enumerate(training_keys), total=total_steps):
        if max_steps != None and i > max_steps: break

        audio_chunk = training_data[key].clone()
        with torch.no_grad(): policy_output = policy.augment(audio_chunk, state=policy_states, return_seed=True, return_mask=True)
        augmented_audio_sample = policy_output['augmented_data']
        policy_states = policy_output['state']
        seeds.append(policy_output['seed'])
        masks.append(policy_output['mask'])
        lr_indexes.append(policy_output['selected_lr_indexes'])
        lr = policy_output['selected_lrs']
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = (optim_args['lr'] * lr).item()

        with torch.no_grad():
            out_teacher = asr_model(audio_signal = audio_chunk)
        out_noised = asr_model(audio_signal = augmented_audio_sample)['final_posteriors']

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

    all_logits, logit_count = torch.zeros((1, audio_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, audio_n//4 + seq_len, tokenizer.vocab_size() + 1))


    for key in tqdm(training_keys):
        audio_chunk = training_data[key].clone()
        with torch.no_grad():
            out = asr_model(audio_signal = audio_chunk)
        logits = out['final_posteriors'][0]
        logits = torch.exp(logits) # convert to prob
        ds_len = logits.shape[-2]
        ratio = audio_chunk.shape[-1] / ds_len
        overlap_ds = int(overlap / ratio)
        model_outputs[key] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds}

    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len     

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) 
    out_text = decoder(logits.squeeze())
    out_text = normalizer(out_text)
    new_wer = word_error_rate_detail(hypotheses=[out_text], references=[text], use_cer=True)[0]

    seeds = torch.stack(seeds, 0)
    masks = torch.stack(masks, 0).squeeze(-1)
    lrs = torch.stack(lr_indexes, 0).squeeze(-1)

    return {
        'original_wer': original_wer,
        'updated_wer': new_wer,
        'masks': masks,
        'seeds': seeds,
        'lrs':lrs,
    }

 
        
       









