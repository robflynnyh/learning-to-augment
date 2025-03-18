import argparse
import torch
from omegaconf.omegaconf import OmegaConf

from l2augment.modelling.models import Policy
from l2augment.utils.helpers import load_rl_models, make_color, backward_pass, load_model as load_policy, save_model as save_policy, lmap
from functools import partial

from l2augment.utils.collate_functions import collate_functions_dict

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


    
class CustomDataset(Dataset):
    def __init__(
            self, 
            files, 
            zero_mean=True, 
            standardize_std=True, 
            divide_by_100=False,
            scale=False, 
            clamp_min=-5, 
            clamp_max=5,
            randomize_order=False, # debug
            decrease_measurement='absolute', # percentage or absolute
            load_audio=True,
            cer_weight=0.0,
            wer_weight=1.0,
            set_minus_or_positive=False
        ):
        self.data = sorted(files)
    
        self.zero_mean = zero_mean
        self.standardize_std = standardize_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = scale
        self.randomize_order = randomize_order
        self.decrease_measurement = decrease_measurement
        self.divide_by_100 = divide_by_100
        self.load_audio = load_audio
        self.cer_weight = cer_weight
        self.wer_weight = wer_weight
        self.set_minus_or_positive = set_minus_or_positive

    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def standardize_pipeline(self, rewards):
        if self.set_minus_or_positive:
            rewards[rewards < 0] = -1
            rewards[rewards > 0] = 1

        rewards_mean = rewards.mean(0, keepdim=True)

        if self.zero_mean:
            rewards = rewards - rewards_mean
        if self.divide_by_100:
            rewards = rewards / 100
            
        if self.clamp_min is not None:
            rewards = rewards.clamp(min=self.clamp_min)
        if self.clamp_max is not None:
            rewards = rewards.clamp(max=self.clamp_max)

        if self.standardize_std:
            if rewards.shape[0] > 1 or rewards_std == 0:
                rewards_std = rewards.std(0, keepdim=True) # center mean then clamp then calculate std before standardizing
                rewards = rewards / (rewards_std + 1e-6)
        if self.scale:
            # min -1, max 1 but avoid 0 division
            rewards_min = rewards.min(dim=0, keepdim=True).values
            rewards_max = rewards.max(dim=0, keepdim=True).values
            if rewards_min == rewards_max:
                rewards = torch.zeros_like(rewards)
            else:
                rewards = 2*(rewards - rewards_min)/(rewards_max - rewards_min) - 1

        return rewards

    
    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio'] if self.load_audio else None
            masks = rollout['mask'] # kept in float8 for memory
            decreases = rollout['reward']   

            misc = {}
            if 'probs' in rollout:
                misc['probs'] = rollout['probs']
            if 'eps' in rollout:
                misc['eps'] = rollout['eps']


            if decreases.ndim == 3: has_wer = True
            else: has_wer = False   

            before, after = decreases.chunk(2, dim=-1)


            if self.decrease_measurement == 'absolute':
                rewards = before - after
            elif self.decrease_measurement == 'percentage':
                rewards = ( (before - after) / before )*100
            else:
                raise ValueError(f"Unknown decrease measurement: {self.decrease_measurement} should be 'percentage' or 'absolute'")
            rewards = rewards.squeeze(-1)

            # replace any nan values with 0 
            rewards[torch.isnan(rewards)] = 0 # can happen due to empty reference and hypothesis

            if has_wer:
                cer, wer = rewards.unbind(-1)
                cer, wer = lmap(self.standardize_pipeline, [cer, wer])
                rewards = cer*self.cer_weight + wer*self.wer_weight
            else:
                rewards = self.standardize_pipeline(rewards)

            all_rewards = rewards

            return {
                'reward': all_rewards, # (repeats)
                'masks':masks, # (masks, 1, C, T)
                **({'audio':audio} if self.load_audio else {}),
                **misc
            }
        except Exception as e:
            logger.info(f"Error loading data: {e}")
            return None

    
def get_eval_config(config):
    """creates a config for running the evaluation script using the validation config in the training script"""
    return {'evaluation': config['validation'], 'checkpointing': config['checkpointing']}

def train_policy(
        policy:Policy,
        optim:MADGRAD,
        config:Dict[str, Any],
        dataloader:DataLoader,
    ):  
        device = policy.device

        policy = policy.train()

        prev_val_cer = float('inf')
        prev_state_dict = {k:v.clone() for k,v in policy.state_dict().items()}

        cur_epoch = 0
        max_tolerance = 2
        remaining_tolerance = max_tolerance
        running = True
        max_steps = config['training'].get('max_steps', -1)

        while running:
            
            policy = policy.eval()
            
            if cur_epoch == 0 and config.get('validation', {}).get('default', None) is not None:
                val_cer = config['validation']['default']
            else: 
                val_cer = run_eval(config = get_eval_config(config), policy_net=policy)

            
            wandb.log({'val_cer':val_cer, 'epoch': cur_epoch})
            logger.info(f'val_cer: {val_cer}')

            if (val_cer > prev_val_cer) or torch.isnan(torch.tensor(val_cer)): remaining_tolerance -= 1
            else:
                prev_val_cer = val_cer
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
              
                loss, losses = policy.forward_pass(batch, device)
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
    rollout_directory = os.path.join(config['generation']['save_dir'], split)

    all_rollouts = os.listdir(rollout_directory)
    all_rollouts = [el for el in all_rollouts if el.endswith('.pt')]
    all_rollouts = [os.path.join(rollout_directory, el) for el in all_rollouts]

    return all_rollouts

import shutil, subprocess, time

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
    

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params_in_million = total_params / 1_000_000
    logger.info(make_color(f"Total trainable parameters (policy): {total_params_in_million:.2f} million", 'green'))

    policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])

    for i in range(50):

        generation_path = config['generation']['save_dir']
        # delete folder
        if os.path.exists(generation_path):
            shutil.rmtree(generation_path)
            os.mkdir(generation_path)

        launch_path = "job.sh"
        # change directory to ./launch_scripts
        os.chdir('./launch_scripts')
        result = subprocess.run(['sbatch', launch_path], capture_output=True, text=True)
        logger.info(f"Launched job: {result.stdout}")
        assert result.returncode == 0, f"Error launching job: {result.stderr}"
        os.chdir('..')
        check_job_cmd = "squeue | grep rjf | grep job.sh | wc -l"
        # loop until zero jobs are running
        jobs_finished = False
        while not jobs_finished:
            result = subprocess.run(check_job_cmd, shell=True, capture_output=True, text=True)
            assert result.returncode == 0, f"Error checking job: {result.stderr}"
            if int(result.stdout) == 0: 
                jobs_finished = True
            else:
                logger.info(f"Waiting for jobs to finish: {int(result.stdout)}")
                # wait 10 seconds before checking again
                time.sleep(10)
        

        train_files = prepare_data(config, split='train')

        dataset_config = config.get('dataset', {})
        train_dataset = CustomDataset(train_files, **dataset_config)

        collate_function = collate_functions_dict[config.get('collate_function', 'default')]
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            collate_fn=collate_function,
            num_workers=config['training'].get('num_workers', 12),
            prefetch_factor=config['training'].get('prefetch_factor', 6),
        )


        policy = train_policy(policy, policy_optim, config, train_dataloader)
        save_policy(policy, config)

        logger.info(f'Finished')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)



