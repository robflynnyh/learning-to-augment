import argparse
import torch
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.augmentation import SpecAugment


from l2augment.modelling.models import Policy
from l2augment.rollout.gpu_parellel import gpu_rollout

from tqdm import tqdm
import logging
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader

import pickle
from typing import List, Dict, Any
from madgrad import MADGRAD
import wandb

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
    return policy_net 


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
    
class CustomDataset(Dataset):
    def __init__(
            self, 
            files,
        ):
        self.data = sorted(files)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            file = torch.load(file, weights_only=True)
            audio = file['audio']
            text = file['text']
            return {
                'audio': audio,
                'text': text,
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    
    audio = []
    text = []
    lengths = []

    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        lengths.append(item['audio'].shape[-1])
        text.append(item['text'])

    lengths = torch.tensor(lengths)
    if lengths.min() != lengths.max():
        max_length = lengths.max()
        for i in range(len(audio)):
            cur_len = audio[i].shape[-1]
            if cur_len < max_length:
                diff = max_length - cur_len
                audio[i] = torch.cat([audio[i], torch.zeros(audio[i].shape[0], audio[i].shape[1], diff)], dim=-1)
                
    audio = torch.cat(audio, dim=0)
    
    return {
        'audio': audio,
        'text': text,
        'lengths': lengths
    }

# move to top!
from einops import rearrange, repeat
from functools import partial
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import load_model as load_asr_model, get_model_class
from matplotlib import pyplot as plt
import random

def render_and_save_noise(noise, path='./noise.png'):
    plt.imshow(noise, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
    plt.savefig(path)

def forward_pass(batch, policy, device, augmentation=None, rollout_args={}):
    audio = batch['audio'].to(device)
    text = batch['text']
    lengths = batch['lengths'].to(device)

    rewards = []
    noises = []

    for i in range(6):
        with torch.no_grad(): augmented_audio, noise, noise_probs = policy.augment(audio, sample=True, return_probs=True, lengths=lengths)
        noises.append(noise)
        
        diff = gpu_rollout(
            policy = policy,
            audio_a = audio,
            audio_b = augmented_audio, 
            text = text,
            device = device,
            audio_lengths = lengths,
            **rollout_args
        )
        rewards.append(diff)

    rewards = torch.stack(rewards, dim=0).T.to(device)
    rewards = (rewards - rewards.mean(dim=-1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)

    if random.random() < 0.1: # 10% of the time
        render_and_save_noise(noises[0][0].cpu().numpy(), path='./noise.png')

    noises = torch.stack(noises, dim=0)
    policy_probs = policy(audio)
    policy_probs = rearrange(policy_probs, 'b t (c p) -> b t c p', p=policy.output_dim).log_softmax(dim=-1)
    policy_probs = repeat(policy_probs, 'b t c p -> r b c t p', r=noises.size(0))
    
    noise_indexes = policy.discretize(noises).unsqueeze(-1)
    policy_probs_at_idx = policy_probs.gather(-1, noise_indexes)
    policy_probs_at_idx = rearrange(policy_probs_at_idx, 'r b c t 1 -> b r t c')
  
    if lengths.min() != lengths.max():
        b, t = policy_probs_at_idx.shape[0], policy_probs_at_idx.shape[2]
        pad_mask = torch.arange(t).to(lengths.device) >= lengths[:, None]
        pad_mask = rearrange(pad_mask, 'b t -> b 1 t 1')
        policy_probs_at_idx.masked_fill_(pad_mask, 0)
    
    loss = -1 * (policy_probs_at_idx * rewards.unsqueeze(-1).unsqueeze(-1)).sum() / rewards.size(0)*rewards.size(1)

    return loss, {'loss':loss}
    


def backward_pass(loss, policy, optim):
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),  1.0) 
    optim.step()


def train_policy(
        policy:Policy,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader,
    ):  
        device = config['training']['device']
        policy = policy.train()

        asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
        asr_model_config = asr_model_checkpoint['config']
        asr_model_state_dict = asr_model_checkpoint['model']
        asr_model_class = get_model_class(config = config)
        tokenizer = load_tokenizer()
        partial_load_asr_model_fn = partial(
            load_asr_model_fn,
            load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
            asr_model_state_dict,
        )
        rollout_args = {"load_asr_model_fn": partial_load_asr_model_fn, "tokenizer": tokenizer}

        prev_val_loss = float('inf')
        prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}

        augmentation = None# SpecAugment(n_time_masks=2, n_freq_masks=2, time_mask_param=-1, freq_mask_param=27, min_p=0.05, max_p=0.3)
        cur_epoch = 0
        running = True

        while running:
            val_loss_sum = 0
            val_count = 0
            
            all_val_losses = None

            pbar = tqdm(val_dataloader)
            policy = policy.eval()
            for batch in pbar:
                if batch == None: continue  
                with torch.no_grad():
                    loss, all_losses = forward_pass(batch, policy, device, rollout_args=rollout_args)
                if loss == None: continue

                if all_val_losses == None:
                    all_val_losses = {k:v.item() for k,v in all_losses.items()}
                else:
                    for k,v in all_losses.items():
                        all_val_losses[k] += v.item()

                val_loss_sum += loss.item()
                val_count += 1
                pbar.set_description(desc=f'val_loss: {val_loss_sum/val_count}')

            val_loss = val_loss_sum/val_count
            wandb.log({'val_policy_loss':val_loss, 'epoch': cur_epoch, **{f'val_{k}':v/val_count for k,v in all_val_losses.items()}})
            print(f'val_loss: {val_loss}')

            if (val_loss > prev_val_loss) or torch.isnan(torch.tensor(val_loss)):
                policy.load_state_dict(prev_state_dict)
                print(f'Validation loss increased. Reverting to previous state')
                break

            if cur_epoch >= config['training']['epochs']:
                running = False
                break

            prev_val_loss = val_loss
            prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}

            policy = policy.train()
            pbar = tqdm(dataloader)
            for batch in pbar:
                if batch == None: continue  
                
                loss, losses = forward_pass(batch, policy, device, augmentation=augmentation, rollout_args=rollout_args)
                if loss == None: continue
         
                wandb.log({'policy_loss':loss.item(), 'epoch': cur_epoch, **{k:v.item() for k,v in losses.items()}})
                
                pbar.set_description(desc=f'loss: {loss.item()}')
                backward_pass(loss, policy, optim)

            cur_epoch += 1

            if config['training'].get('tmp_model_save_path', False):
                save_policy(policy, config, save_path=config['training']['tmp_model_save_path'])

        return policy
        
def save_policy(model, config, save_path=None):
    save_path = config['training']['model_save_path'] if save_path is None else save_path

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
    train_pth = config['audio_samples']['train']
    dev_pth = config['audio_samples']['dev']
   
    train_files = os.listdir(train_pth)
    dev_files = os.listdir(dev_pth)

    train_files = [el for el in train_files if el.endswith('.pt')]
    dev_files = [el for el in dev_files if el.endswith('.pt')]
    
    train_files = [os.path.join(train_pth, el) for el in train_files]
    dev_files = [os.path.join(dev_pth, el) for el in dev_files]

    return train_files, dev_files

import time
def main(config):
    wandb.init(project="l2augment")

    policy = load_rl_models(config=config) 
    policy_path = config['training']['model_save_path']
    if os.path.exists(policy_path):
        load_policy(policy, config)
    policy = policy.to(config['training']['device'])
    

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params_in_million = total_params / 1_000_000
    print(f"Total trainable parameters (policy): {total_params_in_million:.2f} million")

    policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])


    train_files, val_files = prepare_data(config)

    train_dataset = CustomDataset(train_files)
    val_dataset = CustomDataset(val_files)

    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4,
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

    policy = train_policy(policy, policy_optim, config, train_dataloader, val_dataloader)
    save_policy(policy, config)


    print(f'Finished')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)



