import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout import  cpu_rollout
from l2augment.modelling.models import Policy
from lcasr.utils.audio_tools import load_json
from l2augment.utils.data import dataset_functions

import os
from os.path import join
import json
import pickle
import random

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0

def load_rl_models(config): 
    policy_net = Policy()
    policy_net = policy_net
    return policy_net

def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    asr_model.flash_attn = False
    return asr_model


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def save_tensors_in_dictionary(dictionary, filename):
    for key in dictionary:
        assert isinstance(dictionary[key], torch.Tensor), f"Value for key {key} is not a tensor"
        dictionary[key] = dictionary[key].cpu().detach()
        torch.save(dictionary[key], f"{filename}_{key}.pt")


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def find_existing_run_wer(directory, id):
    files = os.listdir(directory)
    files = [el for el in files if el.split('_')[0] == str(id)]
    if len(files) > 0:
        file_pth = files[0]
        file = load_dictionary(join(directory, file_pth))
        return file['original_wer']
    return None

def load_policy(model, config):
    save_path = config['training']['model_save_path']
    try:
        # Load the checkpoint
        checkpoint = torch.load(save_path, map_location='cpu')
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])    
        print(f"Model successfully loaded from {save_path}")
        return
    except FileNotFoundError:
        print(f"Model not found at {save_path}")
        return 
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def main(config):


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

    policy_path = config['training']['model_save_path']
    policy_net = Policy()
    if os.path.exists(policy_path):
        load_policy(policy_net, config)


    for i in range(config['steps']):
        index = config['index']*config['steps'] + i


        save_path = config['generation']['save_dir']
        split = {'train':'train', 'val':'dev', 'dev':'dev'}[config['split']]
        save_path = os.path.join(save_path, config['split'])
        
    
        data = dataset_functions['tedlium3_segmented_data'](config['split'])
        if index >= len(data):
            print(f"Index {index} out of range, exiting gracefully")
            return
        utterances = data[index]['process_fn'](data[index])
        print(len(data),'!')
        id = data[index]['id']
        print(len(utterances))
        print(utterances[0])


        for utt_idx, utterance in enumerate(utterances):
            text = utterance['text']
            audio = utterance['spectrogram']
            utt_id = f'{id}_{utt_idx}'

            rollout_fn = partial(cpu_rollout, 
                                load_asr_model_fn = partial_load_asr_model_fn, 
                                tokenizer = tokenizer,
                                text = text
            )      
    

            rewards = []
            mask_list = []
            for i in range(config['repeats']):
                audio_a = audio.clone()
                if policy_net is None:
                    noise = torch.rand_like(audio_a) * torch.rand_like(audio_a) * 2
                    noise = noise.to(torch.float8_e5m2)
                    audio_b = audio_a + noise.to(audio_a.dtype)
                else:
                    audio_b, noise = policy_net.augment(audio_a)
                    noise = noise.to(torch.float8_e5m2)
                    
                prev_cer, u_cer, _ = rollout_fn(
                    policy = None,
                    audio = audio,
                    audio_a = audio_a,
                    audio_b = audio_b,
                )
                reward = torch.stack([prev_cer, u_cer], dim=-1)
                print(reward)
                rewards.append(reward)
                mask_list.append(noise.to(dtype=torch.float8_e5m2))
                #print(reward)

            path = join(save_path, f'{utt_id}.pt')
            print(path)
            if config['save']:
                if os.path.exists(path):
                    prev_data = torch.load(path)
                    torch.save({
                        'reward': torch.cat([prev_data['reward'], torch.stack(rewards)], dim=0),
                        'mask': torch.cat([prev_data['mask'], torch.stack(mask_list)], dim=0),
                        'audio': audio.to(torch.float16),
                    }, path)
                else:
                    torch.save({
                        'reward': torch.stack(rewards),
                        'mask': torch.stack(mask_list),
                        'audio': audio.to(torch.float16),
                    }, path)
      
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    parser.add_argument('--steps', '-steps', type=int, default=5)
    parser.add_argument('--repeats', '-repeats', type=int, default=2)
    parser.add_argument('--split', '-split', type=str, default='train')
    parser.add_argument('--dont_save', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    config['steps'] = args.steps
    config['repeats'] = args.repeats
    config['save'] = not args.dont_save
    config['split'] = args.split
    main(config)




