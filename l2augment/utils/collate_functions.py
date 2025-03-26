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


collate_functions_dict = {
    "default": mask_collate_fn,
    "1dmask": mask_collate_fn,
    "mask_and_audio": mask_and_audio_collate_fn,
    "vae_based_policy" : vae_based_policy_fn,
    "1dmask_and_audio": partial(mask_and_audio_collate_fn, mask_type='1d'),
}