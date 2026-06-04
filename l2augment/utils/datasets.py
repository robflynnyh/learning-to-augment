import re
import os
from os.path import join
import json
import pickle
from typing import List
import torch, torchaudio
import torch.nn.functional as F
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
            ssl_cache_dir=None,
            ssl_feature_mode='on_the_fly',
            ssl_feature_key='ssl_features',
            ssl_feature_alignment='native',
            ssl_bundle='HUBERT_BASE',
            ssl_device='cpu',
            tedlium_base='/store/store4/data/TEDLIUM_release-3/legacy',
            write_ssl_cache=False,
            **kwargs,
        ):
        super().__init__(files, **kwargs)
        self.ssl_cache_dir = ssl_cache_dir
        self.ssl_feature_mode = ssl_feature_mode
        self.ssl_feature_key = ssl_feature_key
        self.ssl_feature_alignment = ssl_feature_alignment
        self.ssl_bundle_name = ssl_bundle
        self.ssl_device = ssl_device
        self.tedlium_base = tedlium_base
        self.write_ssl_cache = write_ssl_cache
        self._ssl_bundle = None
        self._ssl_model = None
        self._stm_cache = {}

        valid_modes = {'on_the_fly', 'cache', 'auto'}
        if self.ssl_feature_mode not in valid_modes:
            raise ValueError(f"ssl_feature_mode must be one of {sorted(valid_modes)}")
        valid_alignments = {'native', 'mask_token'}
        if self.ssl_feature_alignment not in valid_alignments:
            raise ValueError(f"ssl_feature_alignment must be one of {sorted(valid_alignments)}")

    def _cache_path(self, rollout_path):
        if self.ssl_cache_dir is None:
            return None
        split = os.path.basename(os.path.dirname(rollout_path))
        return os.path.join(self.ssl_cache_dir, split, os.path.basename(rollout_path))

    def _load_cached_features(self, rollout_path):
        cache_path = self._cache_path(rollout_path)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        audio_features = cached[self.ssl_feature_key].to(dtype=torch.float16)
        if audio_features.ndim != 2:
            raise ValueError(f"Expected 2D SSL features in {cache_path}, got {tuple(audio_features.shape)}")
        return audio_features, cache_path

    def _load_ssl_model(self):
        if self._ssl_model is None:
            try:
                self._ssl_bundle = getattr(torchaudio.pipelines, self.ssl_bundle_name)
            except AttributeError as exc:
                raise ValueError(f"Unknown torchaudio SSL bundle: {self.ssl_bundle_name}") from exc
            self._ssl_model = self._ssl_bundle.get_model().eval().to(self.ssl_device)
            for param in self._ssl_model.parameters():
                param.requires_grad = False
        return self._ssl_bundle, self._ssl_model

    @staticmethod
    def _parse_stm(stm_path):
        utterances = []
        with open(stm_path, 'r') as handle:
            for line in handle:
                parts = line.strip().split(' ')
                if len(parts) < 7:
                    continue
                _, _, _, start, end, _, *text = parts
                text = ' '.join(text)
                if text == 'ignore_time_segment_in_scoring':
                    continue
                text = re.sub(r'<[^>]*>', '', text)
                utterances.append({'start': float(start), 'end': float(end), 'text': text})
        return utterances

    def _rollout_to_tedlium(self, rollout_path):
        recording_id, utterance_idx = os.path.splitext(os.path.basename(rollout_path))[0].rsplit('_', 1)
        split = os.path.basename(os.path.dirname(rollout_path))
        split_root = os.path.join(self.tedlium_base, split)
        sph_path = os.path.join(split_root, 'sph', f'{recording_id}.sph')
        stm_path = os.path.join(split_root, 'stm', f'{recording_id}.stm')
        return sph_path, stm_path, int(utterance_idx)

    def _load_utterance(self, rollout_path):
        sph_path, stm_path, utterance_idx = self._rollout_to_tedlium(rollout_path)
        if stm_path not in self._stm_cache:
            self._stm_cache[stm_path] = self._parse_stm(stm_path)
        utterances = self._stm_cache[stm_path]
        if utterance_idx >= len(utterances):
            raise IndexError(f"{rollout_path} maps to utterance {utterance_idx}, but {stm_path} has {len(utterances)} utterances")
        return sph_path, stm_path, utterance_idx, utterances[utterance_idx]

    @staticmethod
    def _load_waveform_segment(sph_path, start_s, end_s):
        info = torchaudio.info(sph_path)
        frame_offset = max(0, int(round(start_s * info.sample_rate)))
        num_frames = max(1, int(round((end_s - start_s) * info.sample_rate)))
        waveform, sample_rate = torchaudio.load(sph_path, frame_offset=frame_offset, num_frames=num_frames)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    @staticmethod
    def _align_features(features, target_steps):
        features = features.transpose(1, 2)
        features = F.interpolate(features, size=target_steps, mode='linear', align_corners=False)
        return features.transpose(1, 2).squeeze(0).contiguous()

    def _extract_ssl_features(self, rollout_path, target_steps):
        bundle, model = self._load_ssl_model()
        sph_path, stm_path, utterance_idx, utterance = self._load_utterance(rollout_path)
        waveform, sample_rate = self._load_waveform_segment(sph_path, utterance['start'], utterance['end'])
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        waveform = waveform.to(self.ssl_device)
        lengths = torch.tensor([waveform.size(-1)], dtype=torch.long, device=waveform.device)
        with torch.no_grad():
            extracted = model.extract_features(waveform, lengths=lengths)
        if isinstance(extracted, tuple):
            features, feature_lengths = extracted
        else:
            features, feature_lengths = extracted, None
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if feature_lengths is not None:
            features = features[:, :int(feature_lengths[0].item())]
        if self.ssl_feature_alignment == 'mask_token':
            audio_features = self._align_features(features.cpu(), int(target_steps))
        else:
            audio_features = features.squeeze(0).cpu().contiguous()
        audio_features = audio_features.to(dtype=torch.float16)

        cache_path = self._cache_path(rollout_path)
        if self.write_ssl_cache and cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                self.ssl_feature_key: audio_features,
                'ssl_bundle': self.ssl_bundle_name,
                'rollout_path': rollout_path,
                'sph_path': sph_path,
                'stm_path': stm_path,
                'utterance_idx': utterance_idx,
                'start': utterance['start'],
                'end': utterance['end'],
                'target_steps': int(target_steps),
                'ssl_feature_alignment': self.ssl_feature_alignment,
            }, cache_path)
        return audio_features, cache_path

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if item is None:
            return None
        try:
            cache_result = None
            if self.ssl_feature_mode in {'cache', 'auto'}:
                cache_result = self._load_cached_features(item['source_path'])
            if cache_result is None:
                if self.ssl_feature_mode == 'cache':
                    raise FileNotFoundError(f"Missing SSL cache sidecar: {self._cache_path(item['source_path'])}")
                audio_features, cache_path = self._extract_ssl_features(
                    item['source_path'],
                    target_steps=int(item['generation_length']),
                )
            else:
                audio_features, cache_path = cache_result

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
