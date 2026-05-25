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


class RewardConditionedMaskLMDataset(Dataset):
    def __init__(
            self,
            files,
            reward_metric='wer',
            normalization='per_utterance_minmax',
            degenerate_range_eps=1e-6,
            degenerate_value=0.5,
            logger=print,
        ):
        self.data = sorted(files)
        self.reward_metric = reward_metric
        self.normalization = normalization
        self.degenerate_range_eps = degenerate_range_eps
        self.degenerate_value = degenerate_value
        self.logger = logger

    def __len__(self):
        return len(self.data)

    def _raw_reward(self, reward):
        if reward.ndim == 3:
            metric_to_index = {'cer': 0, 'wer': 1}
            metric_idx = metric_to_index.get(self.reward_metric, self.reward_metric)
            return reward[:, int(metric_idx), 0] - reward[:, int(metric_idx), 1]
        before, after = reward.chunk(2, dim=-1)
        return (before - after).squeeze(-1)

    def _normalize_reward(self, reward):
        reward = reward.to(dtype=torch.float32)
        reward[torch.isnan(reward)] = 0.0
        reward[torch.isinf(reward)] = 0.0

        if self.normalization not in ('per_utterance', 'per_utterance_minmax'):
            raise ValueError(f"Unsupported reward normalization: {self.normalization}")

        reward_min = reward.min()
        reward_range = reward.max() - reward_min
        degenerate = reward.numel() <= 1 or reward_range <= self.degenerate_range_eps
        if degenerate:
            return torch.full_like(reward, float(self.degenerate_value)), True
        return (reward - reward_min) / reward_range, False

    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file)
            generation = rollout['generation'].to(torch.long)
            if generation.ndim == 1:
                generation = generation.unsqueeze(0)

            raw_reward = self._raw_reward(rollout['reward'])
            reward, degenerate = self._normalize_reward(raw_reward)

            return {
                'generation': generation,
                'reward': reward,
                'raw_reward': raw_reward.to(dtype=torch.float32),
                'source_path': file,
                'generation_length': torch.tensor(generation.shape[-1], dtype=torch.long),
                'degenerate_reward_group': torch.tensor(degenerate, dtype=torch.bool),
            }
        except Exception as e:
            self.logger(f"Error loading reward-conditioned mask LM data: {e}")
            return None


class AudioRewardConditionedMaskLMDataset(RewardConditionedMaskLMDataset):
    def __init__(
            self,
            files,
            ssl_cache_dir,
            ssl_feature_key='ssl_features',
            **kwargs,
        ):
        super().__init__(files, **kwargs)
        self.ssl_cache_dir = ssl_cache_dir
        self.ssl_feature_key = ssl_feature_key

    def _cache_path(self, rollout_path):
        split = os.path.basename(os.path.dirname(rollout_path))
        return os.path.join(self.ssl_cache_dir, split, os.path.basename(rollout_path))

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if item is None:
            return None
        try:
            cache_path = self._cache_path(item['source_path'])
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Missing SSL cache sidecar: {cache_path}")
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
            audio_features = cached[self.ssl_feature_key].to(dtype=torch.float16)
            if audio_features.ndim != 2:
                raise ValueError(f"Expected 2D SSL features in {cache_path}, got {tuple(audio_features.shape)}")
            item['audio_features'] = audio_features
            item['audio_feature_length'] = torch.tensor(audio_features.shape[0], dtype=torch.long)
            item['audio_feature_cache_path'] = cache_path
            return item
        except Exception as e:
            self.logger(f"Error loading audio-conditioned mask LM data: {e}")
            return None


dataset_classes_dict['default'] = CustomDataset
dataset_classes_dict['MultiStepDataset'] = MultiStepDataset
dataset_classes_dict['MultiStepFMDataset'] = MultiStepFMDataset
dataset_classes_dict['RewardConditionedMaskLMDataset'] = RewardConditionedMaskLMDataset
dataset_classes_dict['AudioRewardConditionedMaskLMDataset'] = AudioRewardConditionedMaskLMDataset
