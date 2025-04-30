import re
import os
from os.path import join
import json
import pickle
from typing import List
import torch, torchaudio
from typing import Tuple
from torch.utils.data import Dataset
from l2augment.utils.helpers import lmap
from l2augment.utils.data import CustomDataset

dataset_classes_dict = {}


class MultiStepDataset(Dataset):
    def __init__(
            self, 
            files, 
            logger=print
        ):
        self.data = sorted(files)
        self.logger = logger

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    @staticmethod
    def min_max_normalize(tensor):
        return (tensor - tensor.min(dim=-1).values.unsqueeze(-1)) / (tensor.max(dim=-1).values.unsqueeze(-1) - tensor.min(dim=-1).values.unsqueeze(-1))

    def standardize_pipeline(self, rewards):
        rewards = self.min_max_normalize(rewards)
        # set all nan values to 1
        rewards[torch.isnan(rewards)] = 1.
        return rewards


    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio'] 
            generations = rollout['generations'] 
            rewards = rollout['rewards']   
            # replace any nan values with 0 
            rewards[torch.isnan(rewards)] = 0 # can happen due to empty reference and hypothesis
            
            if 'top_idxs' in rollout: paths = rollout['top_idxs']
            else: paths = rewards.topk(1, dim=-1).indices.squeeze(-1)

            rewards = self.standardize_pipeline(rewards)

            results = {
                'audio': audio,
                'generations': generations,
                'rewards': rewards,
                'paths': paths
            }

            if 'entropy' in rollout: results['entropy'] = rollout['entropy']
            if 'n_losses' in rollout: results['n_losses'] = rollout['n_losses']
            
            return results

        except Exception as e:
            self.logger(f"Error loading data: {e}")
            return None


class MultiStepFMDataset(Dataset):
    def __init__(
            self, 
            files, 
            logger=print,
            include_audio=False
        ):
        self.data = sorted(files)
        self.logger = logger
        self.incude_audio = include_audio

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    @staticmethod
    def standardize(tensor):
        rewards_mean = tensor.mean(dim=-1, keepdim=True)
        rewards_std = tensor.std(dim=-1, keepdim=True)
        tensor = tensor - rewards_mean
        if rewards_std.sum() > 0:
            tensor = tensor / (rewards_std + 1e-6)
        return tensor

    def standardize_pipeline(self, rewards):
        rewards = self.standardize(rewards)
        # set all nan values to 1
        rewards[torch.isnan(rewards)] = 0.0
        return rewards

    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio'] 
            masks = rollout['masks'] 
            rewards = rollout['rewards']   
            top_idxs = rollout['top_idxs']
            global_decreases = torch.tensor(rollout['all_decreases']) * 100
            global_decreases[torch.isnan(global_decreases)] = 0 # can happen due to empty reference and hypothesis
            global_decreases = global_decreases.clamp(-1., 1.)
            rollout_repeat = rollout['rollout_repeat']
            global_decrease = global_decreases[rollout_repeat]
            
            # replace any nan values with 0 
            rewards[torch.isnan(rewards)] = 0 # can happen due to empty reference and hypothesis

            rewards = self.standardize_pipeline(rewards)
          
            # audio = {i:v for i,v in enumerate(audio)}
            # generations = {i:v for i,v in enumerate(generations)}
            returns = {
                'masks': masks,
                'rewards': rewards,
                'paths': top_idxs,
                'global_decrease': global_decrease,
            }
            if self.incude_audio:
                returns['audio'] = audio
                
            return returns

        except Exception as e:
            self.logger(f"Error loading data: {e}")
            return None


dataset_classes_dict['default'] = CustomDataset
dataset_classes_dict['MultiStepDataset'] = MultiStepDataset
dataset_classes_dict['MultiStepFMDataset'] = MultiStepFMDataset