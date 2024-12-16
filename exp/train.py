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
from torch_scatter import scatter_logsumexp

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = int(0.875*AUDIO_CHUNK_SIZE_DEFAULT)


from concurrent.futures import ThreadPoolExecutor

def load_pt_file(file_path):
    """
    Load a single PyTorch .pt file
    
    Args:
        file_path (str): Path to the .pt file
    
    Returns:
        Loaded torch object
    """
    return torch.load(file_path)

def load_multiple_pt_files(file_paths, max_workers=None):
    """
    Load multiple .pt files in parallel
    
    Args:
        file_paths (list): List of file paths to load
        max_workers (int, optional): Maximum number of threads to use. 
                                     Defaults to None (auto-determined)
    
    Returns:
        list: Loaded torch objects in the same order as input paths
    """
    #return [load_pt_file(file_path) for file_path in file_paths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(load_pt_file, file_paths))

def load_rl_models(config): 
    policy_net = Policy()
    # prev_policy_net = Policy()
    # value_net = Value()
    # policy_net = policy_net
    return policy_net #, prev_policy_net, value_net

def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
    
reward_cache = {}
class CustomDataset(Dataset):
    def __init__(
            self, 
            files, 
            zero_mean=True, 
            standardize_std=False, 
            scale=True, 
            clamp_min=-5, 
            clamp_max=5,
            randomize_order=False # debug
        ):
        self.data = files
        self.keys = sorted(list(self.data.keys()))
        self.zero_mean = zero_mean
        self.standardize_std = standardize_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = scale
        self.randomize_order = randomize_order

    def __len__(self):
        # Return the total number of samples
        return len(self.keys)
    
    def __getitem__(self, idx):
        try:
            key = self.keys[idx]
            audio_path = self.data[key]['audio']
            audio = torch.load(audio_path)
            masks_and_rewards_paths = self.data[key]['masks_and_rewards']
            masks_and_rewards = torch.load(masks_and_rewards_paths)
            masks = masks_and_rewards['mask']
            rewards = masks_and_rewards['reward']
         

            assert len(masks) == len(rewards), f"Length of masks and rewards not equal for {key}"

            # replace any nan values with 0 
            rewards[torch.isnan(rewards)] = 0 # can happen due to empty reference and hypothesis

            if self.clamp_min is not None:
                rewards = rewards.clamp(min=self.clamp_min)
            if self.clamp_max is not None:
                rewards = rewards.clamp(max=self.clamp_max)

            rewards_mean, rewards_std = rewards.mean(), rewards.std()
        
            if self.zero_mean:
                rewards = rewards - rewards_mean
            if self.standardize_std:
                if rewards.shape[0] > 1 or rewards_std == 0:
                    rewards = rewards / (rewards_std + 1e-6)
            if self.scale:
                # min -1, max 1 but avoid 0 division
                rewards = 2*(rewards - rewards.min())/(rewards.max() - rewards.min() + 1e-6) - 1
                rewards = (rewards + 1) / 2

            if self.randomize_order:
                rewards = rewards[torch.randperm(rewards.shape[0])] # debug

            # z = torch.zeros_like(rewards)
            # z[rewards > 0] = 1
            # rewards = z
        


            return {
                'reward': rewards, # (repeats)
                'masks':masks, # (masks, 2, C, T)
                'audio':audio # (1, C, T)
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    
    audio = []
    masks = []
    rewards = []
    item_idxs = []
    counts = []

    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        masks.append(item['masks'])
        rewards.append(item['reward'])
        item_idxs.extend([i]*item['reward'].shape[0])
        counts.append(item['reward'].shape[0])

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
        'counts': counts
    }



def forward_pass(batch, policy, device):
    masks = batch['masks'].to(device)
    audio = batch['audio'].to(device)
    rewards = batch['rewards'].to(device)
    counts = batch['counts'].to(device)

    x = policy(audio, masks, counts)        

    prediction = x.mean(dim=(1,2)).sigmoid()
    print(prediction[:10])
    print(rewards[:10])
    print('--')
    loss = torch.nn.functional.mse_loss(input=prediction, target=rewards, reduction='mean')
    #loss = torch.nn.functional.binary_cross_entropy(input=prediction, target=rewards, reduction='mean')

    return loss

def backward_pass(loss, policy, optim):
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0) 
    optim.step()

def train_policy(
        policy:Policy,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader
    ):  
        device = config['training']['device']
        policy = policy.train()

        prev_val_loss = float('inf')
        prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}
        
        for epoch in range(config['training']['epochs']):
            val_loss_sum = 0
            val_count = 0
            

            pbar = tqdm(val_dataloader)
            for batch in pbar:
                if batch == None: continue  
                with torch.no_grad():
                    loss = forward_pass(batch, policy, device)
                if loss == None: continue
         
                val_loss_sum += loss.item()
                val_count += 1
                pbar.set_description(desc=f'val_loss: {val_loss_sum/val_count}')

            val_loss = val_loss_sum/val_count
            wandb.log({'val_policy_loss':val_loss, 'epoch': epoch})
            print(f'val_loss: {val_loss}')

            if val_loss > prev_val_loss:
                policy.load_state_dict(prev_state_dict)
                print(f'Validation loss increased. Reverting to previous state')
                break

            prev_val_loss = val_loss
            prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}


            pbar = tqdm(dataloader)
            for batch in pbar:
                if batch == None: continue  
                
                loss = forward_pass(batch, policy, device)
                if loss == None: continue
         
                wandb.log({'policy_loss':loss.item(), 'epoch': epoch})
                
                pbar.set_description(desc=f'loss: {loss.item()}')
                backward_pass(loss, policy, optim)

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

def prepare_data(config):
    rollout_directory = config['generation']['save_dir']
    audio_directory = config['teacher_logits_generation']['save_dir']

    all_rollouts = os.listdir(rollout_directory)
    all_rollouts_masks_and_rewards = [el for el in all_rollouts if 'rewards_and_masks' in el and el.endswith('.pt')]

    all_audio = os.listdir(audio_directory)
    all_audio = [el for el in all_audio if '_audio_' in el and el.endswith('.pt')]

    id_mask_and_reward_pairs = {}


    for file in tqdm(all_rollouts_masks_and_rewards):
        id = file.replace('rewards_and_masks', 'audio')
        id_mask_and_reward_pairs[id] = file

    all_data = {}
    for file in tqdm(all_audio):
        if file in id_mask_and_reward_pairs and file in id_mask_and_reward_pairs:
            audio = join(audio_directory, file)
            masks_and_rewards = join(rollout_directory, id_mask_and_reward_pairs[file])
            
            
            all_data[file] = {
                'audio': audio,
                'masks_and_rewards': masks_and_rewards
            }


    all_recordings = sorted(list(set(["_".join(el.split("_")[:-1]) for el in list(all_data.keys())])))
    assert len(all_recordings) > 2, "Need atleast 2 unique recordings to train the model for train and val split"
    num_val = min(int(len(all_recordings)*0.025), 1)
    val_files = all_recordings[:num_val]
    #train_files = all_recordings[num_val:num_val+100]
    train_files = all_recordings[num_val:]

    all_val_recordings = {el:all_data[el] for el in all_data if "_".join(el.split("_")[:-1]) in val_files}
    all_train_recordings = {el:all_data[el] for el in all_data if "_".join(el.split("_")[:-1]) in train_files}

    return all_train_recordings, all_val_recordings

def main(config):
    wandb.init(project="l2augment")
    
    train_files, val_files = prepare_data(config)
    # if not os.path.exists('train_val_files.pkl'):
    #     train_files, val_files = prepare_data(config)
    #     with open('train_val_files.pkl', 'wb') as file:
    #         pickle.dump([train_files, val_files], file)
    # else:
    #     with open('train_val_files.pkl', 'rb') as file:
    #         train_files, val_files = pickle.load(file)

    train_dataset = CustomDataset(train_files)
    val_dataset = CustomDataset(val_files)

    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=16,
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

    policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])

    policy = train_policy(policy, policy_optim, config, train_dataloader, val_dataloader)

    save_policy(policy, config)
    print(f'Finished')

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)






#   ###################
#         load_policy(policy, config)
#         asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
#         asr_model_config = asr_model_checkpoint['config']
#         asr_model_state_dict = asr_model_checkpoint['model']
#         tokenizer = load_tokenizer()
#         asr_model_class = get_model_class(config = config)

#         partial_load_asr_model_fn = partial(
#             load_asr_model_fn,
#             load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
#             asr_model_state_dict,
#         )
#         dataset = "this_american_life"
#         data = dataset_functions[dataset]("dev")
#         cur_data = data[0]
#         audio_spec, text = cur_data['process_fn'](cur_data)
#         gpu_eval(
#             policy=policy,
#             load_asr_model_fn=partial_load_asr_model_fn,
#             tokenizer=tokenizer,
#             audio=audio_spec,
#             text=text,
#             device = config['training']['device'],
#             max_steps = config['generation']['max_steps'],
#         )
#         ####################
#         #exit()