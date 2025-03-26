import torch
from madgrad import MADGRAD
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
from lcasr.utils.augmentation import SpecAugment
import random
from functools import partial
from einops import repeat, rearrange
import math
from tqdm import tqdm
import functools
import numpy as np
from l2augment.rollout import cpu_rollout as singlestep_rollout


def seed_all(seed):
    """A decorator to temporarily set the seed for random, numpy, and torch."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save current states
            random_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()

            try:
                # Set seeds
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Execute function
                return func(*args, **kwargs)

            finally:
                # Restore original states
                random.setstate(random_state)
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)

        return wrapper
    return decorator



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

def apply_masker(x, method=None):
  n_time_masks = random.randint(3, 16)
  min_p = random.random()/2
  time_masker = SpecAugment(n_time_masks=n_time_masks, n_freq_masks=0, freq_mask_param=0, zero_masking=True, min_p=min_p)
  freq_masks = random.randint(5,7)
  freq_masker = SpecAugment(n_time_masks=0, n_freq_masks=freq_masks, freq_mask_param=34, zero_masking=True)
  method = random.randint(0,2) if method is None else method

  if method == 0:
    x = time_masker(x)
  elif method == 1:
    x = freq_masker(x)
  else:
    x = time_masker(x)
    x = freq_masker(x)
  return x


@seed_all(123456)
def cpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        utterances:List[Dict[str, Any]],
        seq_len:int=2048,
        overlap=0.875,
        optim_args:Dict[str, Any] = {"lr":8e-6},
        original_wer = None,
        max_steps = None,
        shuffle=True,
        augmentation_config:Dict[str, Any] = {},
        epochs = 1,
        **kwargs
    ):
    

    dtype = torch.float32 #temporary
    torch.use_deterministic_algorithms(True, warn_only=True)

    overlap = round(seq_len*overlap)
    # audio = audio.to(dtype=dtype) #temporary
    # audio_n = audio.shape[-1]
    
    # text = normalizer(text)
    
    asr_model = load_asr_model_fn()
    asr_model = asr_model.to(dtype=dtype)
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    asr_model.eval()
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device)
    policy = policy.to(device)

    #original_wer = 0.283
    original_hypothesis = None
    if original_wer is None:
        hyps, refs = [], []
        for utterance in utterances:
            audio_chunk = utterance['spectrogram']
            with torch.no_grad():
                out = asr_model(audio_signal = audio_chunk.to(device))
            logits = out['final_posteriors'][0].to('cpu')
            out_text = decoder(logits)
            out_text = normalizer(out_text)
            hyps.append(out_text)
            refs.append(normalizer(utterance['text']))

        original_wer = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)[0]
        original_hypothesis = out_text
        print(f"Original WER: {original_wer}")


    utterance_mask_pairs = {}
    utterance_lr_pairs = {}

    for i, utterance in enumerate(utterances):
        print(f'Finding top-5 masks for utterance {i+1}/{len(utterances)}')
        cur_audio = utterance['spectrogram'].to(dtype=dtype)
        cur_text = utterance['text']
        masks = []
        rewards = []
        
        lrs = []
        
        for repeat in range(25):
            audio_a = cur_audio.clone()
            # outputs = policy.augment(audio_a, **augmentation_config)
            # audio_b, mask = outputs[0], outputs[1]
            #n_time_masks = random.randint(3, 10)
            #masker = SpecAugment(n_time_masks=6, n_freq_masks=0, freq_mask_param=0, zero_masking=True, min_p=0.2)
            mask = torch.ones_like(audio_a)
            method = random.randint(0,2)
            mask = apply_masker(mask, method=0)
            audio_b = mask * audio_a
            # mask = torch.rand_like(audio_a)
            # mask = torch.bernoulli(mask)
            # audio_b = mask * audio_a
            lr_mult = random.choice([1.0])

            masks.append(mask)

            prev_cer, u_cer, _ = singlestep_rollout(
                policy = None,
                audio = cur_audio,
                audio_a = audio_a,
                audio_b = audio_b,
                load_asr_model_fn = load_asr_model_fn,
                tokenizer = tokenizer,
                text = cur_text,
                optim_args = {"lr":9e-2},
                return_wer = True,
                verbose=False
            )

           
            reward = prev_cer - u_cer
            reward_wer = reward[-1]
         
            rewards.append(reward_wer)
            print(reward_wer.item())
            lrs.append(lr_mult)
            
        # select the top-5 rewards
        rewards = torch.stack(rewards)
        top_rewards, top_indices = torch.topk(rewards, 5, dim=0)
        top_masks = torch.stack([masks[i] for i in top_indices])
        top_lrs = torch.tensor([lrs[i] for i in top_indices])
        utterance_mask_pairs[i] = top_masks
        utterance_lr_pairs[i] = top_lrs
 

    asr_model = load_asr_model_fn()
    asr_model = asr_model.to(dtype=dtype)
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), **optim_args)
    asr_model.eval()


  
    for epoch in range(epochs*8):
        
        utterance_ids = list(range(len(utterances)))
        if shuffle: random.shuffle(utterance_ids)
        for i, utt_id in enumerate(utterance_ids):
            cur_utterance = utterances[utt_id]
            cur_audio = cur_utterance['spectrogram'].to(device, dtype=dtype)
            cur_text = cur_utterance['text']
            top_masks = utterance_mask_pairs[utt_id]
            
            # select a random mask from the top-5 masks
            selection = random.randint(0, top_masks.shape[0]-1)
            cur_mask = top_masks[selection]
            cur_lr = utterance_lr_pairs[utt_id][selection]
            if cur_mask.ndim == 2: cur_mask = cur_mask.unsqueeze(-1)
            augmented_audio_sample = cur_mask * cur_audio

            #print(f'Current lr: {cur_lr.item()}')

            with torch.no_grad():
                out_teacher = asr_model(audio_signal = cur_audio)
            out_noised = asr_model(audio_signal = augmented_audio_sample)['final_posteriors']

            teacher_output_posteriors = out_teacher['final_posteriors'].squeeze(0)
            pseudo_targets = decoder(teacher_output_posteriors) 
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets))[None]
            N, B = out_noised.shape[1], out_noised.shape[0]
            total_tokens_in_loss = N * B
            loss = ctc_loss_fn(out_noised.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * out_noised.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss
            optimizer.zero_grad()
            loss.backward()

            # change the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = optim_args['lr']*cur_lr.item()

            if kwargs.get("clip", False): torch.nn.utils.clip_grad_norm_(asr_model.parameters(), kwargs["clip"]) 

            optimizer.step()

    
    if epochs > 0:
        hyps, refs = [], []
        for utterance in utterances:
            audio_chunk = utterance['spectrogram']
            with torch.no_grad():
                out = asr_model(audio_signal = audio_chunk.to(device))
            logits = out['final_posteriors'][0].to('cpu')
            out_text = decoder(logits)
            out_text = normalizer(out_text)
            hyps.append(out_text)
            refs.append(normalizer(utterance['text']))

        new_wer = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)[0]
    else:
        new_wer = original_wer
    # seeds = torch.stack(seeds, 0)
    # masks = torch.stack(masks, 0).squeeze(-1)
    print(original_wer, new_wer, '-')

    return {
        'original_cer': original_wer,
        'updated_cer': new_wer,
        'hypothesis': out_text,
        'original_hypothesis': original_hypothesis,

    }

 
        
       









