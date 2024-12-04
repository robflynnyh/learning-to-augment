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


def load_rl_models(config): 
    policy_net = Policy(
        input_dim=config['policy']['input_dim'],
        masks_path=config['policy']['masks_path']
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
    def __init__(self, files, rewards_mean:float, rewards_std:float):
        # Convert inputs to torch tensors
        # self.base_path = path
        self.files = files #os.listdir(path)
        self.nbd = True
        self.rewards_mean = rewards_mean
        self.rewards_std = rewards_std

    def __len__(self):
        # Return the total number of samples
        return len(self.files)
    
    def __getitem__(self, idx):
        data = load_dictionary(self.files[idx])
        seeds = data['seeds']
        #lrs = data['lrs']
        masks = data['masks']
        if self.nbd:
            seeds = seeds.transpose(0,1)
            #masks = masks.transpose(0,1)
        reward = data['original_wer'] - data['updated_wer']
        #reward = (reward - self.rewards_mean) / (self.rewards_std + 1e-8)
        
        return {
            'reward': reward,
            'seeds': seeds,
            'masks':masks,
            #'lrs': lrs
        }
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = [item['masks'].size(0) for item in batch]
    max_length = max(lengths)
    lrs = []
    seeds = []
    masks = []
    rewards = []
    for i, item in enumerate(batch):
        item_length = lengths[i]
        lr, seed, mask = None, item['seeds'], item['masks']
        to_pad = max_length - item_length
        if to_pad > 0:
            #lr = torch.cat((lr, torch.zeros(to_pad)), dim=0)
            seed = torch.cat((seed, torch.zeros(1, to_pad, seed.size(-1))), dim=1)
            mask = torch.cat((mask, torch.zeros(to_pad)), dim=1)
        #lrs.append(lr)
        seeds.append(seed)
        masks.append(mask)
        rewards.append(item['reward'])
    #lrs = torch.stack(lrs, dim=0).to(dtype=torch.long)
    masks = torch.stack(masks, dim=0)
    seeds = torch.cat(seeds, dim=0)
    rewards = torch.tensor(rewards)
    lengths = torch.LongTensor(lengths)

    return {
        #'lrs':lrs,
        'masks':masks,
        'seeds':seeds,
        'rewards':rewards,
        'lengths':lengths
    }

path_reward_dict = {}

def get_rewards_dist_data(path_to_rollout_directory:str):
    files = os.listdir(path_to_rollout_directory)
    rewards = []
    for file in tqdm(files, desc="fetching reward data"):
        file_path = join(path_to_rollout_directory, file)
        if file_path in path_reward_dict:
            rewards.append(path_reward_dict[file_path])
        else:
            try:
                cur_dict = load_dictionary(file_path)
                original_wer, updated_wer = cur_dict['original_wer'], cur_dict['updated_wer']
                reward = original_wer - updated_wer
                rewards.append(reward)
                path_reward_dict[file_path] = reward
            except:pass
        
    rewards = torch.tensor(rewards)

    top_100_mean = torch.tensor(sorted(rewards.tolist(), reverse=True)[:100]).mean()
    
    logger.info(f'Average reward: {rewards.mean()}, Max reward: {rewards.max()}, Median: {torch.median(rewards)}, Top 100 Mean: {top_100_mean}')
    

    wandb.log(data={
        'mean_reward': rewards.mean(),
        'max_reward': rewards.max(),
        'median_reward': torch.median(rewards),
        'top_100_mean':top_100_mean
    }, commit=False )

    rewards.clamp_(-0.1, 0.1)    

    rewards_mean =  rewards.mean().item()
    rewards_std = rewards.std().item()
    
    return rewards_mean, rewards_std

def train(
        policy:torch.nn.Module,
        optim:torch.optim.Optimizer,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader
    ):  
        device = policy.device = config['training']['device']
        loss_fn = torch.nn.GaussianNLLLoss(reduction='none')
        prev_val_loss = torch.inf
        policy = policy.eval()
        for epoch in range(config['training']['epochs']):
            total_val_loss = 0
            total_items_in_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                    seeds = batch['seeds'].to(device)
                    masks = batch['masks'].to(device)
                    rewards = batch['rewards'].to(device)
                    lengths = batch['lengths'].to(device)

                    prev_masks = policy.masks[masks]
                    prev_masks_pad = torch.zeros(prev_masks.shape[0], 1, prev_masks.shape[-1], device=prev_masks.device)
                    prev_masks = torch.cat((prev_masks_pad, prev_masks),dim=1)[:,:-1,:]
                 
                    mask_means, mask_stds, x, hn = policy.forward_mask(seeds, prev_mask=prev_masks)
                    print(mask_means.mean((0,1)))
                    print(mask_stds.mean((0,1)))
        
                    selected_mask_means = torch.gather(mask_means, dim=-1, index=masks.unsqueeze(-1)).squeeze(-1)
                    selected_mask_stds = torch.gather(mask_stds, dim=-1, index=masks.unsqueeze(-1)).squeeze(-1)
                   
                    dist = torch.distributions.Normal(loc=selected_mask_means, scale=selected_mask_stds)
                   
                    loss = -dist.log_prob(rewards.unsqueeze(-1))
                    #loss = loss_fn(selected_mask_means, rewards.unsqueeze(-1).expand_as(selected_mask_means))
                    pad_mask = ~(torch.arange(0, lengths.max())[None].repeat(lengths.size(0),1).to(device) < lengths[:, None])
                    total_val_loss += torch.masked_fill(loss, pad_mask, 0.0).sum().item()
                    total_items_in_val_loss += (~pad_mask).sum()
            val_loss = total_val_loss / total_items_in_val_loss
            wandb.log({'val_loss': val_loss},commit=False)

            if (prev_val_loss - val_loss) < 0.0: break
            prev_val_loss = val_loss

            policy = policy.train()
            pbar = tqdm(dataloader)
            for batch in pbar:
                seeds = batch['seeds'].to(device)
                masks = batch['masks'].to(device)
                rewards = batch['rewards'].to(device)
                lengths = batch['lengths'].to(device)
                #lr_indexes = batch['lrs'].to(device)
         
                prev_masks = policy.masks[masks]
                prev_masks_pad = torch.zeros(prev_masks.shape[0], 1, prev_masks.shape[-1], device=prev_masks.device)
                prev_masks = torch.cat((prev_masks_pad, prev_masks),dim=1)[:,:-1,:]
                mask_means, mask_stds, x, hn = policy.forward_mask(seeds, prev_mask=prev_masks)
                selected_mask_means = torch.gather(mask_means, dim=-1, index=masks.unsqueeze(-1)).squeeze(-1)
                selected_mask_stds = torch.gather(mask_stds, dim=-1, index=masks.unsqueeze(-1)).squeeze(-1)
                # print(selected_mask_means)
                # print(rewards)   
                mean_mean = selected_mask_means.mean().item()
                mean_std = selected_mask_stds.mean().item()
                mean_reward = rewards.mean().item()
             
                dist = torch.distributions.Normal(loc=selected_mask_means, scale=selected_mask_stds)
                
                loss = -dist.log_prob(rewards.unsqueeze(-1))
                #loss = loss_fn(selected_mask_means, rewards.unsqueeze(-1).expand_as(selected_mask_means))
                pad_mask = ~(torch.arange(0, lengths.max())[None].repeat(lengths.size(0),1).to(device) < lengths[:, None])
                loss = ((torch.masked_fill(loss, pad_mask, 0.0).sum() / (~pad_mask).sum()) * pad_mask.size(0)) / config['training']['batch_size']
            
                loss_to_log = loss.item()
                wandb.log({'loss':loss_to_log, 'mean_mean_at_t':mean_mean, 'mean_std_at_t':mean_std, 'mean_reward_at_t':mean_reward, 'epoch': epoch})
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
        r_mean, r_std = None, None#get_rewards_dist_data(rollout_directory)
      
        all_files = os.listdir(config['generation']['save_dir'])
        all_files = [join(config['generation']['save_dir'], file) for file in all_files]
        random.shuffle(all_files)
        val_name = all_files[0].split('/')[-1].split('_')[0]
        val_files = [el for el in all_files if el.split('/')[-1].split('_')[0] == val_name]
        train_files = [el for el in all_files if el.split('/')[-1].split('_')[0] != val_name]

        print(val_files)

        train_dataset = CustomDataset(train_files, r_mean, r_std)
        val_dataset = CustomDataset(val_files, r_mean, r_std)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=12,
            prefetch_factor=8
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=4,
            prefetch_factor=2
        )
        policy = load_rl_models(config=config)
        policy = policy.to(config['training']['device'])

        total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params_in_million = total_params / 1_000_000

        print(f"Total trainable parameters: {total_params_in_million:.2f} million")

        policy_optim = MADGRAD(policy.parameters(), lr=config['training']['lr'])

        policy = train(policy, policy_optim, config, train_dataloader, val_dataloader)
        save_policy(policy, config)
        del policy
        command = "squeue | grep rjf | grep job.sh | wc -l"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = int(result.stdout.strip())
       
        # if result < 45:
        #     command = "sbatch job.sh"
        #     subprocess.run(command, shell=True, capture_output=True, text=True)
        #     print(f'---- RESUBMITTED JOB STACK ----')


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




