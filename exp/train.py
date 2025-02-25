import argparse
import torch
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.augmentation import SpecAugment


from l2augment.modelling.models import Policy

from tqdm import tqdm
import logging
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader

import pickle
from typing import List, Dict, Any
from madgrad import MADGRAD
import wandb
from einops import rearrange


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = int(0.875*AUDIO_CHUNK_SIZE_DEFAULT)


import subprocess
def count_slurm_jobs(user_pattern="rjf", job_pattern="job"):
    """
    Count Slurm jobs matching specific patterns.
    
    Args:
        user_pattern (str): Pattern to search for in the job details (default: "rjf")
        job_pattern (str): Additional pattern to search for (default: "job")
    
    Returns:
        int: Number of matching jobs
        
    Raises:
        subprocess.CalledProcessError: If the squeue command fails
        FileNotFoundError: If squeue is not available on the system
    """
    try:
        # Construct the command as a list of arguments
        cmd = f"squeue | grep {user_pattern} | grep {job_pattern} | awk '{{print $1}}' | wc -l"
        
        # Execute the command and capture output
        result = subprocess.check_output(cmd, shell=True, text=True)
        
        # Convert the result to an integer and return
        return int(result.strip())
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        raise
    except FileNotFoundError:
        print("Error: squeue command not found. Are you on a system with Slurm installed?")
        raise

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
            zero_mean=True, 
            standardize_std=True, 
            divide_by_100=False,
            scale=False, 
            clamp_min=-5, 
            clamp_max=5,
            randomize_order=False, # debug
            decrease_measurement='absolute' # percentage or absolute
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

    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio']
            masks = rollout['mask'] # kept in float8 for memory
            decreases = rollout['reward']          

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

            # notnegative = torch.zeros_like(rewards)
            # notnegative[rewards >= 0] = 1

            #rank_rewards = torch.argsort(torch.argsort(rewards, dim=0, descending=True), dim=0, descending=False)
            
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
                #rewards = (rewards + 1) / 2

            # z = torch.zeros_like(rewards)
            # z[rewards > 0] = 1
            # rewards = z
            
            all_rewards = rewards#torch.cat([rewards, notnegative, rank_rewards], dim=-1)

            return {
                'reward': all_rewards, # (repeats)
                'masks':masks, # (masks, 1, C, T)
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



def forward_pass(batch, policy, device, augmentation=None):
    indexes = policy.discretize(batch['masks'].to(device, dtype=torch.float32)).squeeze(1).transpose(-1,-2).unsqueeze(-1)
    rewards = batch['rewards'].to(device)
    audio = batch['audio'].to(device)
    item_idxs = batch['item_idxs'].to(device)

    # add rewards together to go from batch*rollouts num_rewards to batch num_rewards
    rewards = rewards.to(torch.float32)
    out = torch.zeros_like(indexes, dtype=torch.float32).expand(-1, -1, -1, policy.output_dim)
    src = rearrange(rewards, 'r -> r 1 1 1').expand(-1, out.shape[1], out.shape[2], out.shape[3])
    rewards = torch.scatter_add(out, -1, indexes, src)
    out = rearrange(torch.zeros_like(audio), 'b c t -> b t c 1').expand(-1, -1, -1, policy.output_dim)
    out = out.to(torch.float32)
    item_idxs = rearrange(item_idxs, 'idx -> idx 1 1 1').expand_as(rewards)
    rewards = torch.scatter_add(out, 0, item_idxs, rewards)
    rewards = rearrange(rewards, 'b t c p -> b t c p')
    # -

    audio = batch['audio'].to(device)
    
    lengths = batch['lengths'].to(device)

    if augmentation is not None: audio = augmentation(audio)

    pred = policy(audio)     

    pred = rearrange(pred, 'b t (c p) -> b t c p', p=policy.output_dim).log_softmax(dim=-1)

    if lengths.min() != lengths.max():
        b, t, c, p = pred.shape
        pad_mask = torch.arange(t).to(device) >= lengths[:, None]
        pred = torch.masked_fill(pred, pad_mask.unsqueeze(-1).unsqueeze(-1), 0)
    
    loss = -1 * (pred * rewards).sum() / item_idxs.shape[0]
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
                try:
                    if batch == None: continue  
                    with torch.no_grad():
                        loss, all_losses = forward_pass(batch, policy, device)
                    if loss == None: continue

                    if all_val_losses == None:
                        all_val_losses = {k:v.item() for k,v in all_losses.items()}
                    else:
                        for k,v in all_losses.items():
                            all_val_losses[k] += v.item()

                    val_loss_sum += loss.item()
                    val_count += 1
                    pbar.set_description(desc=f'val_loss: {val_loss_sum/val_count}')
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

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
                try:  
                    
                    loss, losses = forward_pass(batch, policy, device, augmentation=augmentation)
                    if loss == None: continue
            
                    wandb.log({'policy_loss':loss.item(), 'epoch': cur_epoch, **{k:v.item() for k,v in losses.items()}})
                    
                    pbar.set_description(desc=f'loss: {loss.item()}')
                    backward_pass(loss, policy, optim)
                except Exception as e:
                    print(f"Error in training: {e}")
                    continue

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

    policy = load_rl_models(config=config) 
    policy_path = config['training']['model_save_path']
    if os.path.exists(policy_path):
        load_policy(policy, config)
    policy = policy.to(config['training']['device'])
    

    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params_in_million = total_params / 1_000_000
    print(f"Total trainable parameters (policy): {total_params_in_million:.2f} million")

    policy_optim = MADGRAD(policy.parameters(), lr=config['policy']['lr'])


    running = True
    while running:
        num_jobs = count_slurm_jobs()
        if num_jobs == 0:
            # run commands sbatch sbatch job_val.sh
            result = subprocess.run(['sbatch', 'job_val.sh'])
            print(f"Submitted job: {result}")
            result = subprocess.run(['sbatch', 'job.sh'])
            print(f"Submitted job: {result}")
            # rerun count_slurm_jobs every 10 seconds
        finished_slurm_jobs = False
        while not finished_slurm_jobs:
            num_jobs = count_slurm_jobs()
            if num_jobs == 0:
                finished_slurm_jobs = True
            else:
                print(f"Waiting for jobs to finish. {num_jobs} jobs running, sleeping for 10 seconds")
                time.sleep(10)


        train_files, val_files = prepare_data(config)

        train_dataset = CustomDataset(train_files)
        val_dataset = CustomDataset(val_files)

        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=12,
            prefetch_factor=12   
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
        running = False


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



