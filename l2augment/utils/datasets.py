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

            paths = rewards.topk(1, dim=-1).indices.squeeze(-1)

            rewards = self.standardize_pipeline(rewards)

            # audio = {i:v for i,v in enumerate(audio)}
            # generations = {i:v for i,v in enumerate(generations)}
            return {
                'audio': audio,
                'generations': generations,
                'rewards': rewards,
                'paths': paths
            }
        except Exception as e:
            self.logger(f"Error loading data: {e}")
            return None



dataset_classes_dict['default'] = CustomDataset
dataset_classes_dict['MultiStepDataset'] = MultiStepDataset