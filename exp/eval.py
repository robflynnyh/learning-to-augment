import argparse
import torch
import torchaudio
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.audio_tools import total_seconds
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.utils.helpers import load_model as load_policy, load_asr_model_fn

from l2augment.rollout.cpu_multistep import cpu_rollout as multistep_rollout
from l2augment.rollout.singlestep import rollout as singlestep_rollout

from l2augment.modelling.models import Policy
from lcasr.utils.audio_tools import load_json
import re
import os
from os.path import join
import json
import pickle
import random
from l2augment.utils.data import dataset_functions
from l2augment.utils.helpers import load_rl_models
from lcasr.eval.wer import word_error_rate_detail

rollout_fns = {
    'singlestep': singlestep_rollout,
    'multistep': multistep_rollout,
    'default': multistep_rollout
}

_SSL_MODEL_CACHE = {}


def make_audio_ssl_feature_fn(config, rec_dict):
    policy_class = config.get('policy', {}).get('class', 'default')
    if policy_class != 'AudioRewardConditionedMaskLM':
        return None

    audio_path = rec_dict.get('audio')
    if not isinstance(audio_path, str):
        raise ValueError(
            "AudioRewardConditionedMaskLM eval currently requires rec_dict['audio'] "
            "to be a single raw audio path"
        )

    dataset_config = config.get('dataset', {})
    bundle_name = dataset_config.get('ssl_bundle', 'HUBERT_BASE')
    ssl_device = dataset_config.get('ssl_device', config.get('training', {}).get('device', 'cuda'))

    def load_ssl_model(device):
        target_device = str(device if ssl_device == 'same' else ssl_device)
        cache_key = (bundle_name, target_device)
        if cache_key not in _SSL_MODEL_CACHE:
            try:
                bundle = getattr(torchaudio.pipelines, bundle_name)
            except AttributeError as exc:
                raise ValueError(f"Unknown torchaudio SSL bundle: {bundle_name}") from exc
            model = bundle.get_model().eval().to(target_device)
            for param in model.parameters():
                param.requires_grad = False
            _SSL_MODEL_CACHE[cache_key] = (bundle, model, torch.device(target_device))
        return _SSL_MODEL_CACHE[cache_key]

    def extract_features(chunk_start_frame, audio_chunk, device):
        bundle, model, target_device = load_ssl_model(device)
        info = torchaudio.info(audio_path)
        start_s = total_seconds(int(chunk_start_frame))
        duration_s = total_seconds(int(audio_chunk.shape[-1]))
        frame_offset = max(0, int(round(start_s * info.sample_rate)))
        num_frames = max(1, int(round(duration_s * info.sample_rate)))
        waveform, sample_rate = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        waveform = waveform.to(target_device)
        lengths = torch.tensor([waveform.size(-1)], dtype=torch.long, device=target_device)
        with torch.no_grad():
            extracted = model.extract_features(waveform, lengths=lengths)
        if isinstance(extracted, tuple):
            features, feature_lengths = extracted
        else:
            features, feature_lengths = extracted, None
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if feature_lengths is not None:
            feature_length = int(feature_lengths[0].item())
            features = features[:, :feature_length]
        else:
            feature_length = int(features.size(1))
        return features.squeeze(0).contiguous(), feature_length

    return extract_features


def main(config, policy_net=None):

    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    
    asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
    asr_model_config = asr_model_checkpoint['config']
    asr_model_state_dict = asr_model_checkpoint['model']

    rollout_function = rollout_fns[config.get('evaluation', {}).get('rollout_fn', 'default')]

    partial_load_asr_model_fn = partial(
        load_asr_model_fn,
        load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
        asr_model_state_dict,
    )

    if policy_net is None:
        policy_net = load_rl_models(config)
        load_policy(policy_net, config)
   
    rollout_fn = partial(rollout_function, 
                         load_asr_model_fn = partial_load_asr_model_fn, 
                         tokenizer = tokenizer, 
                         verbose = False, 
                         original_wer=None
    )
    

    dataset = config.get('evaluation', {}).get('dataset', 'earnings22')
    split = config.get('evaluation', {}).get('split', 'test')
    epochs = config.get('evaluation', {}).get('epochs', 1)
    optim_args = config.get('evaluation', {}).get('optim_args', {"lr":8e-6})
    data = dataset_functions[dataset](split)
    
    indexes = config.get('indexes', [-1])
    indexes = indexes if sum(indexes) != -1 else range(len(data))

    u_hyps, o_hyps, refs = [], [], [] 
    for i, index in enumerate(indexes):
        print(f'Index {i+1}/{len(indexes)} - {index}')
        cur_data = data[index]
        print('---', cur_data['id'], '---')
        audio_spec, gold_text = cur_data['process_fn'](cur_data)
        audio_ssl_feature_fn = make_audio_ssl_feature_fn(config, cur_data)
    
    

        rollout_output = rollout_fn(
            policy = policy_net,
            audio = audio_spec,
            text = gold_text,
            augmentation_config = config.get('evaluation', {}).get('augmentation_config', {}),
            audio_feature_fn = audio_ssl_feature_fn,
            epochs = epochs,
            optim_args = optim_args
        )

        print(rollout_output['original_cer'], rollout_output['updated_cer'])

        u_hyps.append(rollout_output['hypothesis'])
        o_hyps.append(rollout_output['original_hypothesis']) 
        refs.append(rollout_output['reference'])


    cer = config.get('evaluation', {}).get('use_cer', False)
    eval_type = 'WER' if not cer else 'CER'

    original_wer = word_error_rate_detail(hypotheses=o_hyps, references=refs, use_cer=cer)[0]
    print(f"Original {eval_type}: {original_wer}")

    wer = word_error_rate_detail(hypotheses=u_hyps, references=refs, use_cer=cer)[0]
    
    print(f"Updated {eval_type}: {wer}")

    save_path = config.get('evaluation', {}).get('save_path', "")
    if save_path != "" and config.get('save', True):
        id = config.get('evaluation', {}).get('id', "0") 
        results = f"ID: {id} - Dataset: {dataset} - Split: {split} - Epochs: {epochs} - Original_WER: {original_wer} - Updated_WER: {wer}"
        with open(save_path, 'a') as file:
            file.write(results)
            file.write('\n')


    return wer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    parser.add_argument('--dont_save', '-dont_save', action='store_true', help='Do not save the results')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['indexes'] = args.indexes
    config['save'] = not args.dont_save
    main(config)


