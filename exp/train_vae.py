import argparse
import torch
from omegaconf.omegaconf import OmegaConf

from l2augment.modelling.models import SingleStateVariationalAutoEncoder as VariationalAutoEncoder

from tqdm import tqdm
import logging
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import random
import pickle
from typing import List, Dict, Any
from madgrad import MADGRAD
import matplotlib.pyplot as plt
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


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def make_color(text, color):
    colors = { # use neon colors
        'green': '\033[38;5;46m',
        'red': '\033[38;5;196m',
        'blue': '\033[38;5;27m',
        'yellow': '\033[38;5;226m',
        'purple': '\033[38;5;129m',
        'cyan': '\033[38;5;45m',
        'white': '\033[38;5;231m',
        'orange': '\033[38;5;208m',
        'pink': '\033[38;5;198m',
        'black': '\033[38;5;0m',
    }
    assert color in colors, f"Color {color} not found. Choose from {list(colors.keys())}"
    return f"{colors[color]}{text}\033[0m"
    
class CustomDataset(Dataset):
    def __init__(
            self, 
            files, 
        ):
        self.data = sorted(files)


    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio']       

            return {
                'audio': audio,
            }
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:    
    audio = []
    lengths = []

    for i, item in enumerate(batch):
        if item == None: continue
        audio.append(item['audio'])
        lengths.append(item['audio'].shape[-1])

    lengths = torch.tensor(lengths)
    if lengths.min() != lengths.max():
        max_length = lengths.max()
        for i in range(len(audio)):
            cur_len = audio[i].shape[-1]
            if cur_len != lengths.max():
                diff = max_length - cur_len
                audio[i] = torch.cat([audio[i], torch.zeros(audio[i].shape[0], audio[i].shape[1], diff)], dim=-1)

    audio = torch.cat(audio)
    
    return {
        'audio': audio,
        'lengths': lengths
    }


def backward_pass(loss, model, optim):
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0) 
    optim.step()

def get_eval_config(config):
    return {'evaluation': config['validation'], 'checkpointing': config['checkpointing']}

def train_vae(
        vae:VariationalAutoEncoder,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        val_dataloader:DataLoader
    ):  
        device = vae.device

        vae = vae.train()

        prev_val_loss = float('inf')
        prev_state_dict = {k:v.clone() for k,v in vae.state_dict().items()}

        cur_epoch = 0
        max_tolerance = 4
        remaining_tolerance = max_tolerance
        running = True

        while running:
            
            vae = vae.eval()
            val_loss_sum = 0
            val_count = 0
            
            all_val_losses = None

            pbar = tqdm(val_dataloader)
            for batch in pbar:
                with torch.no_grad():
                    loss, all_losses, _ = vae.forward_pass(batch, device)
                    if loss == None: continue
                    val_loss_sum += loss.item()
                    val_count += 1
                    if all_val_losses is None:
                        all_val_losses = {k:v.item() for k,v in all_losses.items()}
                    else:
                        for k,v in all_losses.items():
                            all_val_losses[k] += v.item()
                    pbar.set_description(desc=f'val_loss: {loss.item()}')

            val_loss = val_loss_sum / val_count
            wandb.log({'val_loss':val_loss, 'epoch': cur_epoch, **{k:v/val_count for k,v in all_val_losses.items()}})
            print(f'Validation loss: {val_loss}')

            if (val_loss > prev_val_loss) or torch.isnan(torch.tensor(val_loss)): remaining_tolerance -= 1
            else:
                prev_val_loss = val_loss
                prev_state_dict = {k:v.clone() for k,v in vae.state_dict().items()}
                remaining_tolerance = max_tolerance


            if remaining_tolerance == 0:
                vae.load_state_dict(prev_state_dict)
                print(f'Validation loss increased. Reverting to previous state')
                break

            if cur_epoch >= config['training']['epochs']:
                print(f'Reached max epochs: {cur_epoch}/{config["training"]["epochs"]}')
                break


            vae = vae.train()
            pbar = tqdm(dataloader)
            for batch in pbar:
                if batch == None: continue
                try:  
                    
                    loss, losses, prediction = vae.forward_pass(batch, device)
                    if loss == None: continue

                    if random.random() < 0.025:
                        length = batch['lengths'][0].item()
                        plt.imshow(prediction[0,:,:length].detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='magma', vmin=-1, vmax=1)
                        wandb.log({'prediction': plt})
                        plt.close()
                        plt.imshow(batch['audio'][0,:,:length].detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='magma', vmin=-1, vmax=1)
                        wandb.log({'original': plt})
                        plt.close()


                    wandb.log({'policy_loss':loss.item(), 'epoch': cur_epoch, **{k:v.item() for k,v in losses.items()}})
                    
                    pbar.set_description(desc=f'loss: {loss.item()}')
                    backward_pass(loss, vae, optim)
                except Exception as e:
                    print(f"Error in training: {e}")
                    continue

            cur_epoch += 1

            if config['training'].get('tmp_model_save_path', False):
                save_policy(vae, config, save_path=config['training']['tmp_model_save_path'])

        return vae
        
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


def load_model(model, config):
    save_path = config['training']['model_save_path']
    try:
        # Load the checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])    
        print(make_color(f"Model successfully loaded from {save_path}", 'red'))
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
    rollout_directory_train = os.path.join(config['generation']['save_dir'], 'train')
    rollout_directory_val = os.path.join(config['generation']['save_dir'], 'dev')
   

    all_rollouts_train = os.listdir(rollout_directory_train)
    all_rollouts_val = os.listdir(rollout_directory_val)

    all_rollouts_train = [el for el in all_rollouts_train if el.endswith('.pt')]
    all_rollouts_val = [el for el in all_rollouts_val if el.endswith('.pt')]
    
    all_rollouts_train = [os.path.join(rollout_directory_train, el) for el in all_rollouts_train]
    all_rollouts_val = [os.path.join(rollout_directory_val, el) for el in all_rollouts_val]

    return all_rollouts_train, all_rollouts_val

import time
def main(config):
    wandb.init(project="l2augment")

    vae_config = config.get('vae', {})
    vae = VariationalAutoEncoder(**vae_config.get('model', {}))
    vae_path = config['training']['model_save_path']
    
    if os.path.exists(vae_path):
        load_model(vae, config)

    device = config.get('training',{}).get('device', 'cuda')
    if not torch.cuda.is_available(): device = torch.device('cpu')
    vae.device = device
    vae = vae.to(device)
    

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total_params_in_million = total_params / 1_000_000
    print(make_color(f"Total trainable parameters (vae): {total_params_in_million:.2f} million", 'green'))

    vae_optim = MADGRAD(vae.parameters(), lr=config.get('vae',{}).get('lr', 9e-4))

    train_files, val_files = prepare_data(config)

    dataset_config = config.get('dataset', {})
    train_dataset = CustomDataset(train_files, **dataset_config)
    val_dataset = CustomDataset(val_files, **dataset_config)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=6   
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=2
    )

    

    vae = train_vae(vae, vae_optim, config, train_dataloader, val_dataloader)
    save_policy(vae, config)


    print(f'Finished')

# # import subprocess
# import subprocess
# def get_jobs():
#     # run squeue | grep rjf | grep job | awk '{print $1}' | wc -l
#     cmd = "squeue | grep rjf | grep job | awk '{print $1}' | wc -l"
#     p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
#     (output, err) = p.communicate()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)



