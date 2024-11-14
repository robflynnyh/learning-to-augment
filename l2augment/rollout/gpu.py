import torch
from madgrad import MADGRAD
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
import random

DEFAULT_OPTIMIZER_CLASS = MADGRAD
MAX_UTT_LIMIT = 20 #LIMIT TO 10 FOR NOW


def gpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        audio:Tensor,
        text:List[Dict[str,str]],
        chunk_audio_function:Callable,
        chunk_text_function:Callable,
        tokenizer,
        audio_lengths = None,
        optim_args:Dict[str, Any] = {"lr":3e-5},
        **kwargs
    ):

    dtype = torch.float32 #temporary
    audio = audio.to(dtype=dtype) #temporary

    audio_chunks = chunk_audio_function(spec = audio)
    text_chunks = chunk_text_function(text=text, spectogram_length=audio.shape[-1])
    utterances = len(audio_chunks)
    assert utterances == len(text_chunks)

    indexes = list(range(utterances))[:MAX_UTT_LIMIT] 
    
    if kwargs.get('shuffle', True): random.shuffle(indexes)
    

    asr_model = load_asr_model_fn()
    asr_model = asr_model.to('cuda', dtype=dtype)

    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')
    verbose = kwargs.get("verbose", False)
    asr_model.eval()

    initial_wers = {}
    for current_index in indexes:
        audio_sample = audio_chunks[current_index].to('cuda') 
        text_sample = text_chunks[current_index]
        with torch.no_grad():
            out = asr_model(audio_signal = audio_sample)
        initial_prediction = decoder(out['final_posteriors'][0].detach())
        initial_wer = word_error_rate_detail(hypotheses=[initial_prediction], references=[text_sample])[0]
        if initial_wer == float('inf'): initial_wer = 0.0
        initial_wers[current_index] = initial_wer
    
    rewards = torch.zeros(len(indexes))
    seeds = []
    masks = []

    for i, current_index in enumerate(indexes):
        audio_sample = audio_chunks[current_index].to('cuda') 
        text_sample = text_chunks[current_index]
        with torch.no_grad(): policy_output = policy.augment(audio_sample, return_seed=True, return_mask=True)
        
        augmented_audio_sample = policy_output['augmented_data']
        seeds.append(policy_output['seed'])
        masks.append(policy_output['mask'])

        audio_signal = torch.cat([
            audio_sample,
            augmented_audio_sample         
        ])
        
        out = asr_model(audio_signal = audio_signal)
       
        pseudo_targets = decoder(out['final_posteriors'][0].detach())
        noisy_predictions = out['final_posteriors'][1][None]

        #if verbose: print(text_sample, '   ###   ', pseudo_targets, '----------->', decoder(noisy_predictions[0].detach()))

        pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0)
        N, B = noisy_predictions.shape[1], noisy_predictions.shape[0]
        total_tokens_in_loss = N * B
        
        if verbose: print(pseudo_targets.shape, noisy_predictions.transpose(0,1).shape)
        loss = ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * noisy_predictions.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss
        
        optimizer.zero_grad()
        loss.backward()
        if kwargs.get("clip", False): torch.nn.utils.clip_grad_norm_(asr_model.parameters(), kwargs["clip"]) 
        optimizer.step()

        with torch.no_grad(): updated_out = asr_model(audio_signal = audio_sample)
        
        updated_prediction = decoder(updated_out['final_posteriors'][0].detach())
        updated_wer = word_error_rate_detail(hypotheses=[updated_prediction], references=[text_sample])[0]
        if updated_wer == float('inf'): updated_wer = 0.0
        absolute_wer_reduction = initial_wers[current_index] - updated_wer

        gamma_power = torch.arange(i+1).flip(0)
        reward_factor = 0.9**gamma_power
        reward_at_t = reward_factor * absolute_wer_reduction
        rewards[:reward_at_t.size(0)] += reward_at_t

        if verbose: print(absolute_wer_reduction)

    rewards = rewards
    seeds = torch.cat(seeds, dim=0)
    masks = torch.cat(masks, dim=0).squeeze(-1)
    
    return {
        'rewards': rewards,
        'masks': masks,
        'seeds': seeds
    }

 
        
       










