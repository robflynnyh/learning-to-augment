import torch
from madgrad import MADGRAD
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
import random

DEFAULT_OPTIMIZER_CLASS = MADGRAD

def cpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        audio:Tensor,
        chunk_audio_function:Callable,
        tokenizer,
        optim_args:Dict[str, Any] = {"lr":1e-6},
        **kwargs
    ):

    audio_chunks = chunk_audio_function(spec = audio)
    encoded_text_chunks = [torch.LongTensor(tokenizer.encode(text_chunk))[None] for text_chunk in text_chunks]
    encoded_text_chunks_lengths = torch.LongTensor([el.shape[0] for el in encoded_text_chunks])
    assert len(audio_chunks) == len(encoded_text_chunks), 'length mismatch'
    utterances = len(audio_chunks)

    indexes = list(range(utterances))
    
    if kwargs.get('ttt_shuffle', True): random.shuffle(indexes)
    

    asr_model = load_asr_model_fn()
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)
    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')
    verbose = kwargs.get("verbose", False)
    asr_model.eval()

    for current_index in indexes:
        audio_sample = audio_chunks[current_index] 
        augmented_audio_sample = policy.augment(audio_sample)
        audio_signal = torch.cat([
            audio_sample,
            augmented_audio_sample         
        ])

        out = asr_model(audio_signal = audio_signal)
        pseudo_targets = decoder(out['final_posteriors'][0].detach())
        noisy_predictions = out['final_posteriors'][1]

        if verbose: print(pseudo_targets, decoder(noisy_predictions.detach()))

        pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0)
        N, B = noisy_predictions.shape[1], noisy_predictions.shape[0]
        total_tokens_in_loss = N * B

    #     loss = ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * noisy_predictions.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     if kwargs.get("clip", False): torch.nn.utils.clip_grad_norm_(asr_model.parameters(), kwargs["clip"]) 
    #     optimizer.step()

    #     updated_out = asr_model(audio_signal = utt)
    #     updated_predictions = decoder(updated_out['final_posteriors'][0].detach())










