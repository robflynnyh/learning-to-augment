import torch
from typing import List, Dict
from functools import partial

def mask_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    
    masks = []
    rewards = []
    item_idxs = []
    counts = []

    for i, item in enumerate(batch):
        if item == None: continue
        masks.append(item['masks'].to(torch.float16))
        rewards.append(item['reward'])
        item_idxs.extend([i]*item['reward'].shape[0])
        counts.append(item['reward'].shape[0])


    masks = torch.cat(masks, dim=0)
    rewards = torch.cat(rewards, dim=0)
    item_idxs = torch.tensor(item_idxs)
    counts = torch.tensor(counts)
    
    return {
        'masks': masks,
        'rewards': rewards,
        'item_idxs': item_idxs,
        'counts': counts,
    }


def mask_and_audio_collate_fn(batch: List[Dict[str, torch.Tensor]], mask_type='2d') -> Dict[str, torch.Tensor]:    
    assert mask_type in ['1d', '2d'], "mask_type must be either '1d' or '2d'"

    audio = []
    masks = []
    rewards = []
    item_idxs = []
    counts = []
    lengths = []

    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        lengths.append(item['audio'].shape[-1])
        masks.append(item['masks'].to(torch.float16))
        rewards.append(item['reward'])
        item_idxs.extend([i]*item['reward'].shape[0])
        counts.append(item['reward'].shape[0])

    lengths = torch.tensor(lengths)
    if lengths.min() != lengths.max():
        max_length = lengths.max()
        for i in range(len(audio)):
            cur_len = audio[i].shape[-1]
            if cur_len < max_length:
                diff = max_length - cur_len
                audio[i] = torch.cat([audio[i], torch.zeros(audio[i].shape[0], audio[i].shape[1], diff)], dim=-1)
                if mask_type == '2d':
                    masks[i] = torch.cat([masks[i], torch.zeros((masks[i].shape[0], masks[i].shape[1], masks[i].shape[2], diff), dtype=masks[i].dtype)], dim=-1)

                
    audio = torch.cat(audio, dim=0)
    masks = torch.cat(masks, dim=0)
    rewards = torch.cat(rewards, dim=0)
    item_idxs = torch.tensor(item_idxs)
    counts = torch.tensor(counts)
    
    return {
        'audio': audio,
        'masks': masks,
        'rewards': rewards,
        'item_idxs': item_idxs,
        'counts': counts,
        'lengths': lengths
    }

def vae_based_policy_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    

    audio = []
    masks = []
    eps = []
    probs = []
    rewards = []
    item_idxs = []
    counts = []
    lengths = []
    ds_lengths = []

    total_reward = 0


    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        lengths.append(item['audio'].shape[-1])
        masks.append(item['masks'].to(torch.float16))
        rewards.append(item['reward'])
        if 'probs' in item:
            probs.append(item['probs'].to(torch.float16))
        if 'eps' in item:
            eps.append(item['eps'].to(torch.float16))
            ds_lengths.append(item['eps'].shape[-1])
        item_idxs.extend([i]*item['reward'].shape[0])
        counts.append(item['reward'].shape[0])
        total_reward += item['total_reward']

    lengths = torch.tensor(lengths)
    if len(ds_lengths) > 0: ds_lengths = torch.tensor(ds_lengths)
    if lengths.min() != lengths.max():
        max_length = lengths.max()
        for i in range(len(audio)):
            cur_len = audio[i].shape[-1]
            if cur_len < max_length:
                diff = max_length - cur_len
                audio[i] = torch.cat([audio[i], torch.zeros(audio[i].shape[0], audio[i].shape[1], diff)], dim=-1)
                masks[i] = torch.cat([masks[i], torch.zeros((masks[i].shape[0], masks[i].shape[1], masks[i].shape[2], diff), dtype=masks[i].dtype)], dim=-1)
                if len(probs) > 0:
                    probs[i] = torch.cat([probs[i], torch.zeros((probs[i].shape[0], probs[i].shape[1], probs[i].shape[2], diff), dtype=probs[i].dtype)], dim=-1)
            
            if len(eps) > 0:
                if eps[i].shape[-1] < ds_lengths.max():
                    diff = ds_lengths.max() - eps[i].shape[-1]
                    eps[i] = torch.cat([eps[i], torch.zeros(eps[i].shape[0], eps[i].shape[1], eps[i].shape[2], diff)], dim=-1)
            

    audio = torch.cat(audio, dim=0)
    masks = torch.cat(masks, dim=0)
    rewards = torch.cat(rewards, dim=0)
    item_idxs = torch.tensor(item_idxs)
    counts = torch.tensor(counts)
    if len(probs) > 0:
        probs = torch.cat(probs, dim=0)
    if len(eps) > 0:
        eps = torch.cat(eps, dim=0)
    avg_reward = total_reward / sum(counts)

    optional = {}
    if len(probs) > 0:
        optional['probs'] = probs
    if len(eps) > 0:
        optional['eps'] = eps
        optional['ds_lengths'] = ds_lengths

    return {
        'audio': audio,
        'masks': masks,
        'rewards': rewards,
        'item_idxs': item_idxs,
        'counts': counts,
        'lengths': lengths,
        'ds_lengths': ds_lengths,
        'avg_reward': avg_reward,
        **optional
    }


def DTLM_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    

    audio = []
    generations = []
    rewards = []
    item_idxs = []
    counts = []
    lengths = []
    ds_lengths = []


    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        lengths.append(item['audio'].shape[-1])
        rewards.append(item['reward'])
        generations.append(item['generation'].to(torch.long))
        ds_lengths.append(item['generation'].shape[-1])
        item_idxs.extend([i]*item['reward'].shape[0])
        counts.append(item['reward'].shape[0])

    lengths = torch.tensor(lengths)
    ds_lengths = torch.tensor(ds_lengths)

    if lengths.min() != lengths.max():
        max_length = lengths.max()
        for i in range(len(audio)):
            cur_len = audio[i].shape[-1]
            if cur_len < max_length:
                diff = max_length - cur_len
                audio[i] = torch.cat([audio[i], torch.zeros(audio[i].shape[0], audio[i].shape[1], diff)], dim=-1)
       
            if generations[i].shape[-1] < ds_lengths.max():
                diff = ds_lengths.max() - generations[i].shape[-1]
                generations[i] = torch.cat([generations[i], torch.zeros(generations[i].shape[0], diff, dtype=generations[i].dtype)], dim=-1)
        

    audio = torch.cat(audio, dim=0)
    rewards = torch.cat(rewards, dim=0)
    item_idxs = torch.tensor(item_idxs)
    counts = torch.tensor(counts)
    generations = torch.cat(generations, dim=0)

    return {
        'audio': audio,
        'generations': generations,
        'ds_lengths': ds_lengths,
        'rewards': rewards,
        'item_idxs': item_idxs,
        'counts': counts,
        'lengths': lengths,
        'ds_lengths': ds_lengths,
    }


def MultiStep_DTLM_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    

    audio = {}
    generations = {}
    audio_lengths = {}
    generation_lengths = {}
    paths = {}
    counts = {}
    rewards = {}
    batch_idxs = {}
    dropped_idxs = {}
    
    entropy = {}
    n_losses = {}

    step_counts = [len(item['audio']) for item in batch if item is not None]
    max_steps = max(step_counts)

    for step_idx in range(max_steps):
        for i, item in enumerate(batch):
            if item == None: continue
            cur_total_steps = len(item['audio'])
            if step_idx < cur_total_steps:
                if step_idx not in audio: audio[step_idx] = []
                if step_idx not in generations: generations[step_idx] = []
                if step_idx not in audio_lengths: audio_lengths[step_idx] = []
                if step_idx not in generation_lengths: generation_lengths[step_idx] = []
                if step_idx not in paths: paths[step_idx] = []
                if step_idx not in counts: counts[step_idx] = []
                if step_idx not in rewards: rewards[step_idx] = []
                if step_idx not in batch_idxs: batch_idxs[step_idx] = []
                if step_idx not in dropped_idxs: dropped_idxs[step_idx] = []
                if step_idx not in entropy and 'entropy' in item: entropy[step_idx] = []
                if step_idx not in n_losses and 'n_losses' in item: n_losses[step_idx] = []

                audio[step_idx].append(item['audio'][step_idx])
                generations[step_idx].append(item['generations'][step_idx].to(torch.long))
                audio_lengths[step_idx].append(item['audio'][step_idx].shape[-1])
                generation_lengths[step_idx].append(item['generations'][step_idx].shape[-1])
                paths[step_idx].append(item['paths'][step_idx])

                if 'entropy' in item:
                    entropy[step_idx].append(item['entropy'][step_idx])
                if 'n_losses' in item:
                    n_losses[step_idx].append(item['n_losses'][step_idx])

                counts[step_idx].append(item['rewards'][step_idx].shape[0])
                rewards[step_idx].append(item['rewards'][step_idx])
                batch_idxs[step_idx].append(i)

        batch_idxs[step_idx] = torch.tensor(batch_idxs[step_idx])       
        if step_idx != 0:
            prev_batch_idxs = batch_idxs[step_idx-1]
            diff_mask = torch.isin(prev_batch_idxs, batch_idxs[step_idx], invert=True)
            diff = prev_batch_idxs[diff_mask] # get the indexes that are not in the current batch
            indices = torch.arange(prev_batch_idxs.shape[0])[diff_mask]
            dropped_idxs[step_idx] = indices
        else:
            dropped_idxs[step_idx] = torch.tensor([])
           

        audio_lengths[step_idx] = torch.tensor(audio_lengths[step_idx])
        generation_lengths[step_idx] = torch.tensor(generation_lengths[step_idx])
        paths[step_idx] = torch.tensor(paths[step_idx])
        counts[step_idx] = torch.tensor(counts[step_idx])
        rewards[step_idx] = torch.cat(rewards[step_idx], dim=0)
        if 'entropy' in item:
            entropy[step_idx] = torch.stack(entropy[step_idx], dim=0)
        if 'n_losses' in item:
            n_losses[step_idx] = torch.stack(n_losses[step_idx], dim=0)


        if audio_lengths[step_idx].min() != audio_lengths[step_idx].max(): # pad audio to max length
            max_length = audio_lengths[step_idx].max()
            for i in range(len(audio[step_idx])):
                cur_len = audio[step_idx][i].shape[-1]
                if cur_len < max_length:
                    diff = max_length - cur_len
                    audio[step_idx][i] = torch.cat([audio[step_idx][i], torch.zeros(audio[step_idx][i].shape[0], audio[step_idx][i].shape[1], diff)], dim=-1)

        if generation_lengths[step_idx].min() != generation_lengths[step_idx].max():
            max_length = generation_lengths[step_idx].max()
            for i in range(len(generations[step_idx])):
                cur_len = generations[step_idx][i].shape[-1]
                if cur_len < max_length:
                    diff = max_length - cur_len
                    generations[step_idx][i] = torch.cat([generations[step_idx][i], torch.zeros(generations[step_idx][i].shape[0], diff, dtype=generations[step_idx][i].dtype)], dim=-1)

        audio[step_idx] = torch.cat(audio[step_idx], dim=0)
        generations[step_idx] = torch.stack(generations[step_idx], dim=0)

    

    return {
        'audio': audio,
        'generations': generations,
        'audio_lengths': audio_lengths,
        'generation_lengths': generation_lengths,
        'paths': paths,
        'counts': counts,
        'rewards': rewards,
        'dropped_idxs': dropped_idxs,
        'entropy': entropy,
        'n_losses': n_losses,
    }


def MultiStep_FM_ranker_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    

    audio = {}
    masks = {}
    audio_lengths = {}
    paths = {}
    counts = {}
    rewards = {}
    batch_idxs = {}
    dropped_idxs = {}

    step_counts = [len(item['masks']) for item in batch if item is not None]
    has_audio = all('audio' in item for item in batch)
    max_steps = max(step_counts)
    all_global_decreases = torch.tensor([item['global_decrease'] for item in batch if item is not None])

    for step_idx in range(max_steps):
        for i, item in enumerate(batch):
            if item == None: continue
            cur_total_steps = len(item['masks'])
            if step_idx < cur_total_steps:
                if step_idx not in audio and has_audio: audio[step_idx] = []
                if step_idx not in audio_lengths: audio_lengths[step_idx] = []
                if step_idx not in paths: paths[step_idx] = []
                if step_idx not in counts: counts[step_idx] = []
                if step_idx not in rewards: rewards[step_idx] = []
                if step_idx not in batch_idxs: batch_idxs[step_idx] = []
                if step_idx not in dropped_idxs: dropped_idxs[step_idx] = []
                if step_idx not in masks: masks[step_idx] = []

                if has_audio:
                    audio[step_idx].append(item['audio'][step_idx])
                    audio_lengths[step_idx].append(item['audio'][step_idx].shape[-1])

                masks[step_idx].append(item['masks'][step_idx].to(torch.long))
                paths[step_idx].append(item['paths'][step_idx])

                counts[step_idx].append(item['rewards'][step_idx].shape[0])
                rewards[step_idx].append(item['rewards'][step_idx])
                batch_idxs[step_idx].append(i)

        batch_idxs[step_idx] = torch.tensor(batch_idxs[step_idx])       
        if step_idx != 0:
            prev_batch_idxs = batch_idxs[step_idx-1]
            diff_mask = torch.isin(prev_batch_idxs, batch_idxs[step_idx], invert=True)
            diff = prev_batch_idxs[diff_mask] # get the indexes that are not in the current batch
            indices = torch.arange(prev_batch_idxs.shape[0])[diff_mask]
            dropped_idxs[step_idx] = indices
        else:
            dropped_idxs[step_idx] = torch.tensor([])
        

        paths[step_idx] = torch.tensor(paths[step_idx])
        counts[step_idx] = torch.tensor(counts[step_idx])
        rewards[step_idx] = torch.cat(rewards[step_idx], dim=0)
        masks[step_idx] = torch.stack(masks[step_idx], dim=0)

        if has_audio: 
            audio_lengths[step_idx] = torch.tensor(audio_lengths[step_idx])

            if audio_lengths[step_idx].min() != audio_lengths[step_idx].max(): # pad audio to max length
                max_length = audio_lengths[step_idx].max()
                for i in range(len(audio[step_idx])):
                    cur_len = audio[step_idx][i].shape[-1]
                    if cur_len < max_length:
                        diff = max_length - cur_len
                        audio[step_idx][i] = torch.cat([audio[step_idx][i], torch.zeros(audio[step_idx][i].shape[0], audio[step_idx][i].shape[1], diff)], dim=-1)

            audio[step_idx] = torch.cat(audio[step_idx], dim=0)

    returns = {
        'masks': masks,
        'paths': paths,
        'counts': counts,
        'rewards': rewards,
        'dropped_idxs': dropped_idxs,
        'global_decrease': all_global_decreases
    }
    if has_audio:
        returns['audio'] = audio
        returns['audio_lengths'] = audio_lengths

    return returns

collate_functions_dict = {
    "MultiStep_FM_ranker": MultiStep_FM_ranker_fn,
    "MultiStep_DTLM": MultiStep_DTLM_fn,
    "DTLM": DTLM_fn,
    "default": mask_collate_fn,
    "1dmask": mask_collate_fn,
    "mask_and_audio": mask_and_audio_collate_fn,
    "vae_based_policy" : vae_based_policy_fn,
    "1dmask_and_audio": partial(mask_and_audio_collate_fn, mask_type='1d'),
}