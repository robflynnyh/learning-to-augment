import argparse
import torch
from omegaconf.omegaconf import OmegaConf

from l2augment.modelling.models import Policy
from l2augment.utils.helpers import load_rl_models, make_color, backward_pass, load_model as load_policy, save_model as save_policy, lmap
from functools import partial

from l2augment.utils.collate_functions import collate_functions_dict
from l2augment.utils.datasets import dataset_classes_dict

from tqdm import tqdm
import logging
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

import pickle
from typing import List, Dict, Any
from madgrad import MADGRAD
import wandb
from eval import main as run_eval

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

load_policy = partial(load_policy, log_command=logger.info)
save_policy = partial(save_policy, log_command=logger.info)

def get_eval_config(config):
    """creates a config for running the evaluation script using the validation config in the training script"""
    return {'evaluation': config['validation'], 'checkpointing': config['checkpointing']}

def train_policy(
        policy:Policy,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
        dev_dataloader:DataLoader
    ):  
        device = policy.device

        policy = policy.train()

        prev_val_cer = float('inf')
        prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}

        cur_epoch = 0
        max_tolerance = config['training'].get('tolerance', 2)
        remaining_tolerance = max_tolerance
        running = True
        max_steps = config['training'].get('max_steps', -1)

        val_losses = []
        while running:
            
            policy = policy.eval()
            
            pbar = tqdm(dev_dataloader)
            for i, batch in enumerate(pbar):
                if batch == None: continue
              
                loss, losses = policy.forward_pass(batch, device)
                if loss == None: continue
                val_losses.append(loss.item())
        
                wandb.log({'policy_loss':loss.item(), 'epoch': cur_epoch, **{k:v.item() for k,v in losses.items()}})
                
                pbar.set_description(desc=f'loss: {loss.item()}')

            avg_val_loss = sum(val_losses) / len(val_losses)


            
            wandb.log({'avg_val_loss':avg_val_loss, 'epoch': cur_epoch})
            logger.info(f'avg_val_loss: {avg_val_loss}')

            if (avg_val_loss > prev_val_cer) or torch.isnan(torch.tensor(avg_val_loss)): remaining_tolerance -= 1
            else:
                prev_val_cer = avg_val_loss
                prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}
                remaining_tolerance = max_tolerance

            if remaining_tolerance == 0:
                policy.load_state_dict(prev_state_dict)
                logger.info(f'Validation loss increased. Reverting to previous state')
                break

            if cur_epoch >= config['training']['epochs']:
                logger.info(f'Reached max epochs: {cur_epoch}/{config["training"]["epochs"]}')
                break


            policy = policy.train()
            pbar = tqdm(dataloader)
            for i, batch in enumerate(pbar):
                if batch == None: continue
              
                loss, losses = policy.forward_pass(batch, device, wandb=wandb)
                if loss == None: continue
        
                wandb.log({'policy_loss':loss.item(), 'epoch': cur_epoch, **{k:v.item() for k,v in losses.items()}})
                
                pbar.set_description(desc=f'loss: {loss.item()}')
                backward_pass(loss, policy, optim)

                if max_steps != -1 and i >= max_steps:
                    break

            cur_epoch += 1

            if config['training'].get('tmp_model_save_path', False):
                save_policy(policy, config, save_path=config['training']['tmp_model_save_path'])

        return policy
    
def prepare_data(config, split='train'):
    rollout_dirs = config['generation']['save_dir']
    if isinstance(rollout_dirs, str): rollout_dirs = [rollout_dirs]

    all_rollouts = []
    for rollout_dir in rollout_dirs:
        rollout_directory = os.path.join(rollout_dir, split)

        all_rollouts_cur = os.listdir(rollout_directory)
        all_rollouts_cur = [el for el in all_rollouts_cur if el.endswith('.pt')]
        all_rollouts_cur = [os.path.join(rollout_directory, el) for el in all_rollouts_cur]

        all_rollouts.extend(all_rollouts_cur)

    return all_rollouts

def main(config):
    wandb.init(project="l2augment")

    policy = load_rl_models(config=config) 
    policy_path = config['training']['model_save_path']
    if os.path.exists(policy_path):
        load_policy(policy, config)

    device = config.get('training',{}).get('device', 'cuda')
    if not torch.cuda.is_available(): device = torch.device('cpu')
    policy.device = device
    policy = policy.to(device)
    

    total_params = policy.total_parameters()
    total_params_in_million = total_params / 1_000_000
    logger.info(make_color(f"Total trainable parameters (policy): {total_params_in_million:.2f} million", 'green'))

    policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])

    train_files = prepare_data(config, split='train')
    dev_files = prepare_data(config, split='dev')

    dataset_config = config.get('dataset', {})
    
    dataset_class = dataset_classes_dict[config.get('dataset_class', 'default')]
    train_dataset = dataset_class(train_files, **dataset_config, logger=logger.info)
    dev_dataset = dataset_class(dev_files, **dataset_config, logger=logger.info)

    collate_function = collate_functions_dict[config.get('collate_function', 'default')]
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_function,
        num_workers=config['training'].get('num_workers', 12),
        prefetch_factor=config['training'].get('prefetch_factor', 6),
    )

    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_function,
        num_workers=config['training'].get('num_workers', 12),
        prefetch_factor=config['training'].get('prefetch_factor', 6),
    )



    policy = train_policy(policy, policy_optim, config, train_dataloader, dev_dataloader)
    save_policy(policy, config)

    logger.info(f'Finished')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)



