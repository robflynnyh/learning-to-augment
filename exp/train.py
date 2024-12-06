import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout import  cpu_rollout
from l2augment.rollout.gpu_eval import gpu_eval
from einops import rearrange
from l2augment.modelling.models import Policy, Value
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
from l2augment.utils.data import prepare_chunks
from l2augment.utils.data import dataset_functions
from typing import Tuple

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = int(0.875*AUDIO_CHUNK_SIZE_DEFAULT)


def load_rl_models(config): 
    policy_net = Policy()
    prev_policy_net = Policy()
    value_net = Value()
    policy_net = policy_net
    return policy_net, prev_policy_net, value_net

def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    

class CustomDataset(Dataset):
    def __init__(self, files, load_data_function, mean, std):
        # Convert inputs to torch tensors
        # self.base_path = path
        files = ["_".join(el.split('_')[:-1]) for el in files] 
        files = list(set(files))
        self.files = files #os.listdir(path)
        self.nbd = True
        self.load_data_function = load_data_function
        self.mean = mean
        self.std = std

    def __len__(self):
        # Return the total number of samples
        return len(self.files)
    
    def __getitem__(self, idx):
        file_id = int(self.files[idx].split('/')[-1].split('_')[0])
        cur_audio_data = self.load_data_function[file_id]
        audio_spec, _ = cur_audio_data['process_fn'](cur_audio_data, frame_offset=0, num_frames=16000*60*20) # for now
      
        training_data, training_keys = prepare_chunks(audio_spec, AUDIO_CHUNK_SIZE_DEFAULT, AUDIO_CHUNK_OVERLAP_DEFAULT)
        
        mask_path = self.files[idx] + '_masks.pt'
        reward_path = self.files[idx] + '_rewards.pt'
     
        masks = torch.load(mask_path)
        num_steps = masks.size(0)
        training_data = list(training_data.values())[:num_steps]
        #print([el.shape for el in training_data])
        training_data = torch.cat(training_data, dim=0)
        #print(training_data.size(), masks.size())     
        reward = torch.load(reward_path)
        reward = reward

        return {
            'reward': reward,
            'masks':masks,
            'data': training_data
            #'lrs': lrs
        }
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = [item['masks'].size(0) for item in batch]
    assert min(lengths) == max(lengths), 'padding not implemented'
    
    masks = []
    rewards = []
    datas = []
    for i, item in enumerate(batch):
        mask, data, reward = item['masks'], item['data'], item['reward']
        datas.append(data)
        masks.append(mask)
        rewards.append(item['reward'])
    #lrs = torch.stack(lrs, dim=0).to(dtype=torch.long)
    masks = torch.stack(masks, dim=0)
    datas = torch.stack(datas, dim=0)
    rewards = torch.stack(rewards)
    
    return {
        'masks':masks,
        'rewards':rewards,
        'data':datas,
    }

path_reward_dict = {}

def get_rewards_dist_data(path_to_rollout_directory:str):
    files = os.listdir(path_to_rollout_directory)
 
    files = [el for el in files if el.endswith('_rewards.pt')]
    assert len(files) > 0, "No reward files found in the directory, need to run job.sh to generate rollouts once before starting training"
    rewards = []
    for file in tqdm(files, desc="fetching reward data"):
        file_path = join(path_to_rollout_directory, file)
        if file_path in path_reward_dict:
            rewards.append(path_reward_dict[file_path])
        else:
            try:
                reward = torch.load(file_path).flatten()
                rewards.append(reward)
                path_reward_dict[file_path] = reward
            # except all exceptions other than keyboard interrupt
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                if e == KeyboardInterrupt:
                    raise e

    rewards = torch.cat(rewards, dim=0)

    top_100_mean = torch.tensor(sorted(rewards.tolist(), reverse=True)[:100]).mean()
    
    logger.info(f'                                    \n\
                Average reward: {rewards.mean()},     \n\
                Max reward: {rewards.max()},          \n\
                Min reward: {rewards.min()},          \n\
                Median: {torch.median(rewards)},      \n\
                Top 100 Mean: {top_100_mean}          \n\
    ')
    

    wandb.log(data={
        'mean_reward': rewards.mean(),
        'min_reward': rewards.min(),
        'max_reward': rewards.max(),
        'median_reward': torch.median(rewards),
        'top_100_mean':top_100_mean
    }, commit=False )

    rewards_mean =  rewards.mean().item()
    rewards_std = rewards.std().item()
    
    return rewards_mean, rewards_std

def train_value(
        value:Value,
        policy:Policy,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader
    ):  
        
        device = config['training']['device']
        loss_fn = torch.nn.MSELoss(reduction='mean')
        prev_val_loss = torch.inf
        value = value.train()
        
        for epoch in range(config['training']['epochs']):
            total_val_loss = 0
            total_items_in_val_loss = 0

            pbar = tqdm(val_dataloader)
            for batch in pbar:
                masks = batch['masks'].to(device)
                rewards = batch['rewards'].to(device)
                data = batch['data'].to(device)

                masks = policy.masks[masks]
                masks = rearrange(masks, 'b s t maskset c -> b s (maskset c) t')

                with torch.no_grad():
                    p_reward_g_state, p_reward_g_state_n_mask = value.forward_parallel(data, masks)
                loss_a = loss_fn(p_reward_g_state, rewards)
                loss_b = loss_fn(p_reward_g_state_n_mask, rewards)
                loss = loss_a + loss_b
                total_val_loss += loss.item()
                total_items_in_val_loss += 1
                pbar.set_description(desc=f'val_loss: {loss.item()}')
            val_loss = total_val_loss / total_items_in_val_loss
            wandb.log({'val_loss':val_loss, 'epoch': epoch})
            if val_loss > prev_val_loss:
                break
            prev_val_loss = val_loss

            pbar = tqdm(dataloader)
            for batch in pbar:
                masks = batch['masks'].to(device)
                rewards = batch['rewards'].to(device)
                data = batch['data'].to(device)

                
                #mask_probs = policy.forward_parallel(data, masks)
                p_reward_g_state, p_reward_g_state_n_mask = value.forward_parallel(data, masks)
                #masks = masks.transpose(-1,-2) # (B, S, C, T) -> (B, S, T, C)
                #log_probs = (masks*mask_probs + (~masks)*(1-mask_probs)).log()
                #log_probs_at_i = log_probs.sum(dim=(-1,-2)

                loss_a = loss_fn(p_reward_g_state, rewards)
                loss_b = loss_fn(p_reward_g_state_n_mask, rewards)
                loss = loss_a + loss_b
                # loss_to_log = loss.item()
                wandb.log({'value_loss':loss.item(),'value_loss_a':loss_a.item(), 'value_loss_b':loss_b.item(), 'epoch': epoch})
                
                # pbar.set_description(desc=f'loss: {loss_to_log}')
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0) 
                optim.step()

        return value


def train_policy(
        policy:Policy,
        prev_policy:Policy,
        value:Value,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader
    ):  
        
        device = config['training']['device']
        prev_val_loss = torch.inf
        policy = policy.train()
        value = value.eval()

        for epoch in range(config['training']['epochs']):
            total_val_loss = 0
            total_items_in_val_loss = 0

            pbar = tqdm(val_dataloader)
            for batch in pbar:
                masks = batch['masks'].to(device)
                data = batch['data'].to(device)
                rewards = batch['rewards'].to(device)
                p_reward = rewards#.sum(-1)
                
                p_reward = torch.nn.functional.avg_pool1d(p_reward, kernel_size=13, stride=1, padding=6, count_include_pad=False)
             
                # with torch.no_grad():
                #     p_reward_g_state, p_reward_g_state_n_mask = value.forward_parallel(data, masks)
                # p_reward = (p_reward_g_state_n_mask - p_reward_g_state).sum(-1)
       
                masks = policy.masks[masks]
                masks = rearrange(masks, 'b s t maskset c -> b s (maskset c) t')

                mask_probs = policy.forward_parallel(data, masks)
                print(mask_probs.shape)
                tmasks = masks.transpose(-1,-2) # (B, S, C, T) -> (B, S, T, C)
                log_probs = (tmasks*mask_probs + (~tmasks)*(1-mask_probs)).log()
                log_probs_at_i = log_probs.sum(-1)
                
        
                loss = -(log_probs_at_i * p_reward).mean()
                
                pbar.set_description(desc=f'val_loss: {loss.item()}')
               
                total_val_loss += loss.item()
                total_items_in_val_loss += 1

            val_loss = total_val_loss / total_items_in_val_loss
            wandb.log({'val_loss_policy':val_loss, 'epoch': epoch})
            if val_loss > prev_val_loss:
                break
            prev_val_loss = val_loss

            pbar = tqdm(dataloader)
            for batch in pbar:
                masks = batch['masks'].to(device)
                data = batch['data'].to(device)
                rewards = batch['rewards'].to(device)

                p_reward = rewards#.sum(-1)
                p_reward = torch.nn.functional.avg_pool1d(p_reward, kernel_size=13, stride=1, padding=6, count_include_pad=False)
                # with torch.no_grad():
                #     p_reward_g_state, p_reward_g_state_n_mask = value.forward_parallel(data, masks)
                # p_reward = (p_reward_g_state_n_mask - p_reward_g_state).sum(-1)

                masks = policy.masks[masks]
                masks = rearrange(masks, 'b s t maskset c -> b s (maskset c) t')

                mask_probs = policy.forward_parallel(data, masks)
                tmasks = masks.transpose(-1,-2) # (B, S, C, T) -> (B, S, T, C)
                log_probs = (tmasks*mask_probs + (~tmasks)*(1-mask_probs)).log()
                log_probs_at_i = log_probs.sum(-1)
                
        
                loss = -(log_probs_at_i * p_reward).mean()
            
            
                wandb.log({'policy_loss':loss.item(), 'epoch': epoch})
                
                pbar.set_description(desc=f'loss: {loss.item()}')
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

def load_value(model, config):
    save_path = config['value']['save_path']
    try:
        # Load the checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])    
        print(f"Model successfully loaded from {save_path}")
        return
    except FileNotFoundError:
        return 
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_policy(model, config):
    save_path = config['training']['model_save_path']
    try:
        # Load the checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])    
        print(f"Model successfully loaded from {save_path}")
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def save_value(model, config):
    save_path = config['value']['save_path']
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


def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    asr_model.flash_attn = False
    return asr_model

def main(config):
    wandb.init(project="l2augment")
    rollout_directory = config['generation']['save_dir']

    dataset = "this_american_life"
    data_loading_fn = dataset_functions[dataset]("train")

    while True:
        r_mean, r_std = get_rewards_dist_data(rollout_directory)
      
        all_files = os.listdir(config['generation']['save_dir'])
        all_files = [join(config['generation']['save_dir'], file) for file in all_files]
        random.shuffle(all_files)
        val_name = all_files[0].split('/')[-1].split('_')[0]
        val_files = [el for el in all_files if el.split('/')[-1].split('_')[0] == val_name]
        train_files = [el for el in all_files if el.split('/')[-1].split('_')[0] != val_name]

        train_dataset = CustomDataset(train_files, load_data_function=data_loading_fn, mean=r_mean, std=r_std)
        val_dataset = CustomDataset(val_files, load_data_function=data_loading_fn, mean=r_mean, std=r_std)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=8,
            prefetch_factor=4
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=4,
            prefetch_factor=2
        )

        
        policy, _, value = load_rl_models(config=config)
        prev_policy = None
        # loaded = load_policy(prev_policy, config)
        # if loaded == False: 
        #     prev_policy.load_state_dict(policy.state_dict())

        policy = policy.to(config['training']['device'])
        value = value.to(config['training']['device'])
        #prev_policy = prev_policy.to(config['training']['device'])

        ####################
        # load_policy(policy, config)
        # asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
        # asr_model_config = asr_model_checkpoint['config']
        # asr_model_state_dict = asr_model_checkpoint['model']
        # tokenizer = load_tokenizer()
        # asr_model_class = get_model_class(config = config)

        # partial_load_asr_model_fn = partial(
        #     load_asr_model_fn,
        #     load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
        #     asr_model_state_dict,
        # )
        # dataset = "this_american_life"
        # data = dataset_functions[dataset]("dev")
        # cur_data = data[0]
        # audio_spec, text = cur_data['process_fn'](cur_data)
        # gpu_eval(
        #     policy=policy,
        #     load_asr_model_fn=partial_load_asr_model_fn,
        #     tokenizer=tokenizer,
        #     audio=audio_spec,
        #     text=text,
        #     device = config['training']['device'],
        #     max_steps = config['generation']['max_steps'],
        # )
        # ####################
        # #exit()

        total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params_in_million = total_params / 1_000_000

        print(f"Total trainable parameters: {total_params_in_million:.2f} million")

        policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])
        value_optim = MADGRAD(value.parameters(), lr=config['training']['lr'])
        #optims = (policy_optim, value_optim)

        value = train_value(value, policy, value_optim, config, train_dataloader, val_dataloader)
        #policy = train_policy(policy, prev_policy, value, policy_optim, config, train_dataloader, val_dataloader)
        save_value(value, config)
        save_policy(policy, config)


        del policy
        del value
        command = "squeue | grep rjf | grep job.sh | wc -l"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = int(result.stdout.strip())
       
        if result < 40:
            command = "sbatch job.sh"
            subprocess.run(command, shell=True, capture_output=True, text=True)
            subprocess.run(command, shell=True, capture_output=True, text=True)
            print(f'---- RESUBMITTED JOB STACK ----')


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




