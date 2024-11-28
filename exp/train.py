import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout import  cpu_rollout
from l2augment.modelling.models import Policy
from lcasr.utils.audio_tools import load_json
from tqdm import tqdm
import logging
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import random
from typing import List, Dict, Any
from madgrad import MADGRAD
import subprocess
import wandb

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0
POLICY_OUTPUT_DIM_DEFAULT = 80

def load_rl_models(config): 
    policy_net = Policy(
        input_dim=config['policy']['input_dim'],
        output_dim=config['policy'].get('output_dim', POLICY_OUTPUT_DIM_DEFAULT)
    )
    policy_net = policy_net
    return policy_net

def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    

class CustomDataset(Dataset):
    def __init__(self, path, rewards_mean:float, rewards_std:float):
        # Convert inputs to torch tensors
        self.base_path = path
        self.files = os.listdir(path)
        self.nbd = True
        self.rewards_mean = rewards_mean
        self.rewards_std = rewards_std

    def __len__(self):
        # Return the total number of samples
        return len(self.files)
    
    def __getitem__(self, idx):
        data = load_dictionary(join(self.base_path, self.files[idx]))
        seeds = data['seeds']
        lrs = data['lrs']
        masks = data['masks']
        if self.nbd:
            seeds = seeds.transpose(0,1)
            masks = masks.transpose(0,1)
        reward = data['original_wer'] - data['updated_wer']
        #reward = (reward - self.rewards_mean) / (self.rewards_std + 1e-8)
        
        return {
            'reward': reward,
            'seeds': seeds,
            'masks':masks,
            'lrs': lrs
        }
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = [item['lrs'].size(0) for item in batch]
    max_length = max(lengths)
    lrs = []
    seeds = []
    masks = []
    rewards = []
    for i, item in enumerate(batch):
        item_length = lengths[i]
        lr, seed, mask = item['lrs'], item['seeds'], item['masks']
        to_pad = max_length - item_length
        if to_pad > 0:
            lr = torch.cat((lr, torch.zeros(to_pad)), dim=0)
            seed = torch.cat((seed, torch.zeros(1, to_pad, seed.size(-1))), dim=1)
            mask = torch.cat((mask, torch.zeros(1, to_pad, mask.size(-1))), dim=1)
        lrs.append(lr)
        seeds.append(seed)
        masks.append(mask)
        rewards.append(item['reward'])
    lrs = torch.stack(lrs, dim=0).to(dtype=torch.long)
    masks = torch.cat(masks, dim=0)
    seeds = torch.cat(seeds, dim=0)
    rewards = torch.tensor(rewards)
    lengths = torch.LongTensor(lengths)

    return {
        'lrs':lrs,
        'masks':masks,
        'seeds':seeds,
        'rewards':rewards,
        'lengths':lengths
    }

def get_rewards_dist_data(path_to_rollout_directory:str):
    files = os.listdir(path_to_rollout_directory)
    rewards = []
    for file in tqdm(files, desc="fetching reward data"):
        try:
            file_path = join(path_to_rollout_directory, file)
            cur_dict = load_dictionary(file_path)
            original_wer, updated_wer = cur_dict['original_wer'], cur_dict['updated_wer']
            reward = original_wer - updated_wer
            rewards.append(reward)
        except:pass
    rewards = torch.tensor(rewards)

    top_100_mean = torch.tensor(sorted(rewards.tolist(), reverse=True)[:100]).mean()
    rewards.clamp_(-0.1, 0.1)
    logger.info(f'Average reward: {rewards.mean()}, Max reward: {rewards.max()}, Median: {torch.median(rewards)}, Top 100 Mean: {top_100_mean}')
    

    wandb.log(data={
        'mean_reward': rewards.mean(),
        'max_reward': rewards.max(),
        'median_reward': torch.median(rewards),
        'top_100_mean':top_100_mean
    }, commit=False )
    
    return rewards.mean().item(), rewards.std().item()


def train(
        policy:torch.nn.Module,
        optim:torch.optim.Optimizer,
        config:Dict[str, Any],
        dataloader:DataLoader
    ):  
        device = policy.device = config['training']['device']
        for epoch in range(config['training']['epochs']):
            pbar = tqdm(dataloader)
            for batch in pbar:
                seeds = batch['seeds'].to(device)
                masks = batch['masks'].to(device)
                rewards = batch['rewards'].to(device)
                lengths = batch['lengths'].to(device)
                lr_indexes = batch['lrs'].to(device)

                mask_probs, lr_probs, _ = policy.forward(seeds)
                selected_lr_probs = torch.gather(lr_probs, dim=-1, index=lr_indexes.unsqueeze(-1)).squeeze(-1)
                pad_mask = ~(torch.arange(0, lengths.max())[None].repeat(lengths.size(0),1).to(device) < lengths[:, None])
                selected_lr_probs = selected_lr_probs.masked_fill(pad_mask, value=1.0).log()
                total_lr_probs = torch.sum(selected_lr_probs, dim=-1)
                
                mask_prob_at_i = (masks*mask_probs + (1-masks)*(1-mask_probs))
                mask_prob_at_i = torch.masked_fill(mask_prob_at_i, pad_mask[:,:,None], 1.0)
                log_mask_prob_at_i = mask_prob_at_i.log()
              
                entropy = -(mask_prob_at_i*log_mask_prob_at_i).mean()
                total_prob_of_mask = torch.sum(log_mask_prob_at_i, dim=(-1, -2))

                total_probs = total_prob_of_mask + total_lr_probs
                loss = (-total_probs) * rewards
                loss = loss.sum() / config['training']['batch_size']
            
                loss_to_log = loss.item()
                wandb.log({'loss':loss_to_log, 'mask_entropy':entropy.item()})
                pbar.set_description(desc=f'loss: {loss_to_log}')
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0) 
                optim.step()

        return policy

        
def save_policy(model, config):
    save_path = config['training']['model_save_path']
    isnan = False
    for name, param in model.state_dict().items():
        if torch.isnan(param).any():
            isnan = True
    if isnan == False:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, save_path)
        
        logger.info(f"Model saved successfully to {save_path}")
    else:
        logger.info(f"Model not saved due to NaN in weights!")

def main(config):
    wandb.init(project="l2augment")
    rollout_directory = config['generation']['save_dir']

    while True:
        r_mean, r_std = get_rewards_dist_data(rollout_directory)
        dataset = CustomDataset(config['generation']['save_dir'], r_mean, r_std)
        dataloader = DataLoader(
            dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=4,
            prefetch_factor=2
        )
        policy = load_rl_models(config=config)
        policy = policy.to(config['training']['device'])

        policy_optim = MADGRAD(policy.parameters(), lr=config['training']['lr'])

        policy = train(policy, policy_optim, config, dataloader)
        save_policy(policy, config)
        del policy
        command = "squeue | grep rjf | grep job.sh | wc -l"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = int(result.stdout.strip())
        if result < 45:
            command = "sbatch job.sh"
            subprocess.run(command, shell=True, capture_output=True, text=True)
            print(f'---- RESUBMITTED JOB STACK ----')


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




