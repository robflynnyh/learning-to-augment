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
from l2augment.utils.helpers import load_asr_model_fn as create_load_asr_model_fn
from lcasr.utils.general import load_model as load_asr_model, get_model_class
import contextlib

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


@contextlib.contextmanager
def seed_all_ctx(seed):
    """A context manager to temporarily set the seed for random, numpy, and torch."""
    # Save current states
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    torch_cuda_state = None
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.random.get_rng_state_all()
    
    try:
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        yield
    
    finally:
        # Restore original states
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available() and torch_cuda_state is not None:
            torch.cuda.random.set_rng_state_all(torch_cuda_state)


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
def cpu_rollout_presearch(
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
        search_repeats = 5,
        **kwargs
    ):
    raise NotImplementedError("Need to update this function to use policy properly, cba rn as mainly using cpu_rollout_search so adding a NotImplementedError for now")
    
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
        original_hypothesis = hyps
        print(f"Original WER: {original_wer}")


    utterance_mask_pairs = {}
    utterance_lr_pairs = {}

    for i, utterance in enumerate(utterances):
        print(f'Finding top-5 masks for utterance {i+1}/{len(utterances)}')
        cur_audio = utterance['spectrogram'].to(dtype=dtype)
        cur_text = utterance['text']
        masks = []
        rewards = []
        
        
        for repeat in range(search_repeats):
            audio_a = cur_audio.clone()
            # outputs = policy.augment(audio_a, **augmentation_config)
            # audio_b, mask = outputs[0], outputs[1]
            #n_time_masks = random.randint(3, 10)
            #masker = SpecAugment(n_time_masks=6, n_freq_masks=0, freq_mask_param=0, zero_masking=True, min_p=0.2)
            mask = torch.ones_like(audio_a)
            
            mask = apply_masker(mask)
            audio_b = mask * audio_a
            # mask = torch.rand_like(audio_a)
            # mask = torch.bernoulli(mask)
            # audio_b = mask * audio_a

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
            
        # select the top-5 rewards
        rewards = torch.stack(rewards)
        top_rewards, top_indices = torch.topk(rewards, 5, dim=0)
        top_masks = torch.stack([masks[i] for i in top_indices])
        utterance_mask_pairs[i] = top_masks
 

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
        'hypothesis': hyps,
        'original_hypothesis': original_hypothesis,
        'reference': refs

    }

def get_cross_loss(logits:List[Tensor], decoder):
    sentences = [decoder(logit.squeeze(0)) for logit in logits]
    losses = []
  
    for j, logit in enumerate(logits):
     
            encoded = decoder.tokenizer.encode("")
            encoded = torch.LongTensor(encoded)[None]
            input_lengths = torch.LongTensor([logit.shape[1]])
            target_lengths = torch.LongTensor([encoded.shape[1]])
            loss = torch.nn.functional.ctc_loss(
                log_probs=logit.transpose(0, 1),
                targets=encoded,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=logit.shape[-1] - 1,
                reduction='mean',
            )
            losses.append(loss.item())

    losses = torch.tensor(losses)
    mean_cross_loss = losses.mean()
    std_cross_loss = losses.std()
    return mean_cross_loss, std_cross_loss

def find_top_masks(
        audio:Tensor,
        text:str,
        tokenizer,
        policy:Module,
        load_asr_model_fn:Callable,
        decoder,
        repeats=20,
        other_audio=None,
        single_step_lr=4e-2,
        augmentation_args:Dict[str, Any] = {},
        random_path=False,
    ):
    masks = []
    rewards = []
    generations = []
    masked_predictions = []
    for repeat in range(repeats):
        audio_a = audio.clone()
        # outputs = policy.augment(audio_a, **augmentation_config)
        # audio_b, mask = outputs[0], outputs[1]
        #n_time_masks = random.randint(3, 10)
        #masker = SpecAugment(n_time_masks=6, n_freq_masks=0, freq_mask_param=0, zero_masking=True, min_p=0.2)
        #mask = torch.ones_like(audio_a)
        # method = random.randint(0,2)
        # mask = apply_masker(mask)
        policy_out = policy.augment(audio_a, **augmentation_args)
        audio_b, mask = policy_out[0], policy_out[1]
        if len(policy_out) > 2: misc = policy_out[2]
        else: misc = None

        if misc is not None:
            if 'generation' in misc:
                generations.append(misc['generation'])
        
        # mask = torch.rand_like(audio_a)
        # mask = torch.bernoulli(mask)
        # audio_b = mask * audio_a
        masks.append(mask)

        prev_cer, u_cer, _ = singlestep_rollout(
            policy = None,  
            audio = audio if other_audio is None else other_audio,
            audio_a = audio_a,
            audio_b = audio_b,
            load_asr_model_fn = load_asr_model_fn,
            tokenizer = tokenizer,
            text = text,
            optim_args = {"lr":single_step_lr},
            return_wer = True,
            return_masked_prediction=False,
            verbose=False,
            augmented_target=False if other_audio is None else True,
        )
        #masked_predictions.append(masked_prediction)

        
        reward = prev_cer - u_cer
        reward_wer = reward[-1]
        
        rewards.append(reward_wer)
        print(reward_wer.item())
    
    
    # select the top-5 rewards
    rewards = torch.stack(rewards)
    top_rewards, top_indices = torch.topk(rewards, 1, dim=0, largest=True)
    if random_path:
        top_indices = [torch.tensor(random.randint(0, len(masks)-1))]
    top_mask = masks[top_indices[0]]
    if len(generations) > 0:
        generations = torch.stack(generations, 0)

    masks = torch.stack(masks)

    #mean_cross_loss, std_cross_loss = get_cross_loss(masked_predictions, decoder)
    #top_mask = torch.ones_like(top_mask)

    return top_mask, rewards, masks, generations, top_indices[0].item() #, mean_cross_loss, std_cross_loss

def calculate_entropy_from_log_probs(log_probs: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Calculate the entropy from log probabilities.

    Args:
        log_probs (torch.Tensor): Log-probabilities from the model. Expected shape: [..., vocab_size].
        dim (int): The dimension over which to compute entropy (usually vocab dimension).
        keepdim (bool): Whether to retain the reduced dimension.

    Returns:
        torch.Tensor: Entropy values with the same shape as `log_probs` minus the `dim` dimension (unless keepdim=True).
    """
    probs = log_probs.exp()  # Convert log probs to probs
    entropy = -torch.sum(probs * log_probs, dim=dim, keepdim=keepdim)  # Element-wise entropy
    return entropy.mean()


def shuffle_data(data, seed=123456):
    with seed_all_ctx(seed=seed):
        random.shuffle(data)
    return data

def cpu_rollout_search(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        utterances:List[Dict[str, Any]],
        asr_model_config:Dict[str, Any],
        asr_model_class:Module,
        seq_len:int=2048,
        overlap=0.875,
        optim_args:Dict[str, Any] = {"lr":8e-6, 'single_step_lr': 9e-2},
        original_wer = None,
        shuffle=True,
        augmentation_config:Dict[str, Any] = {},
        epochs = 1,
        search_repeats = 5,
        shuffle_seed = 123456,
        random_path=False,
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

    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model.parameters(), lr=optim_args['lr'])
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
        original_hypothesis = hyps
        print(f"Original WER: {original_wer}")



    asr_model.eval()

    masks = []
    audio = []
    rewards = []
    generations = []
    top_idxs = []
    
    n_losses = []
    entropy = []

    for epoch in range(epochs):
        
        utterance_ids = list(range(len(utterances)))
        other_ids = list(range(len(utterances)))
        if shuffle: shuffle_data(utterance_ids, seed=shuffle_seed+epoch)
        random.shuffle(other_ids)
        for i, utt_id in enumerate(utterance_ids):
            print(f'Epoch {epoch+1}/{epochs} - Utterance {i+1}/{len(utterances)} - ID {utt_id}')
            cur_utterance = utterances[utt_id]
            cur_audio = cur_utterance['spectrogram'].to(device, dtype=dtype)
            cur_text = cur_utterance['text']
            
            #other_audio = utterances[other_ids[i]]['spectrogram'].to(device, dtype=dtype)
            #other_text = utterances[other_ids[i]]['text']

            prev_state_dict = asr_model.state_dict()
            prev_optimizer_state_dict = optimizer.state_dict()

            tmp_load_asr_model_fn = partial(
                create_load_asr_model_fn,
                load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
                asr_model.state_dict(),
            )

            #print(asr_model.state_dict()['layers.5.attend.fn.qkv_proj.weight'])

            cur_mask, all_rewards, all_masks, generation, top_idx = find_top_masks(
                cur_audio, 
                cur_text,
                tokenizer, 
                repeats=search_repeats, 
                load_asr_model_fn=tmp_load_asr_model_fn, 
                policy=policy,
                single_step_lr=optim_args.get('single_step_lr', 9e-2),
                augmentation_args=augmentation_config,
                random_path=random_path,
                decoder=decoder,
                #other_audio=utterances[other_ids[i]]['spectrogram'].to(device, dtype=dtype),
            )
            #cur_mask = apply_masker(torch.ones_like(cur_audio))
                            

            masks.append(all_masks)
            rewards.append(all_rewards)
            audio.append(cur_audio.clone().to('cpu'))
            if len(generation) > 0:
                generations.append(generation)
            top_idxs.append(top_idx)

            asr_model.load_state_dict(prev_state_dict)
            optimizer.load_state_dict(prev_optimizer_state_dict)    

            #print(asr_model.state_dict()['layers.5.attend.fn.qkv_proj.weight'])
            
            if cur_mask.ndim == 2: cur_mask = cur_mask.unsqueeze(-1)
            augmented_audio_sample = cur_mask * cur_audio

            #print(f'Current lr: {cur_lr.item()}')

            with torch.no_grad():
                out_teacher = asr_model(audio_signal = cur_audio)
                out_random = asr_model(audio_signal = torch.randn_like(cur_audio, device=device))
            out_noised = asr_model(audio_signal = augmented_audio_sample)['final_posteriors']

            teacher_output_posteriors = out_teacher['final_posteriors'].squeeze(0)

            entropy.append(calculate_entropy_from_log_probs(teacher_output_posteriors))

            pseudo_targets = decoder(teacher_output_posteriors) 
            pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets))[None]
            N, B = out_noised.shape[1], out_noised.shape[0]
            total_tokens_in_loss = N * B
            loss = ctc_loss_fn(out_noised.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * out_noised.shape[0]), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0])) / total_tokens_in_loss


            tgts = torch.LongTensor(tokenizer.encode(""))[None]
            N, B = out_noised.shape[1], out_noised.shape[0]
            n_loss = torch.nn.functional.ctc_loss(
                out_random['final_posteriors'].transpose(0, 1),
                tgts,
                torch.LongTensor([N] * out_noised.shape[0]),
                torch.LongTensor([tgts.shape[1]] * tgts.shape[0]),
                blank=out_noised.shape[-1] - 1,
                reduction='mean',
            )
            n_losses.append(n_loss.item())

            optimizer.zero_grad()
            loss.backward()


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

    rewards = torch.stack(rewards, 0)
    top_idxs = torch.tensor(top_idxs, dtype=torch.long)
    # print(rewards.shape)
    # print([el.shape for el in masks])
    # print([el.shape for el in audio])
    # exit()
    n_losses = torch.tensor(n_losses, dtype=torch.float32)
    entropy = torch.tensor(entropy, dtype=torch.float32)

    # torch.save(
    #     {'n_losses': n_losses, 'entropy': entropy},
    #     './e_0.pt'
    # )
    # exit()

    # exit()

    returns = {
        'original_wer': original_wer,
        'updated_wer': new_wer,
        'hypothesis': hyps,
        'original_hypothesis': original_hypothesis,
        'reference': refs,
        'rewards': rewards,
        'masks': masks,
        'audio': audio,
        'top_idxs': top_idxs,
        'n_losses': n_losses,
        'entropy': entropy
    }
    if len(generations) > 0:
        returns['generations'] = generations
    return returns


        
       









