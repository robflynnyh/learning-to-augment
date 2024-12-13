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
from lcasr.utils.augmentation import SpecAugment

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
        return 
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def main(config):

    for i in range(config['steps']):
        index = config['index']*config['steps'] + i


        save_path = config['generation']['save_dir']
        teacher_logits_dir = config['teacher_logits_generation']['save_dir']
        tokenizer = load_tokenizer()
        asr_model_class = get_model_class(config = config)
    
        files = os.listdir(teacher_logits_dir)
        files = [el for el in files if 'audio' in el]
        files = sorted(files)

        path = files[index].replace('audio', 'rewards_and_masks')
        path = join(save_path, path)

        if os.path.exists(path) and config['save']:
            print(f"File {path} already exists, skipping")
            return

        if index >= len(files):
            print(f"Index {index} out of range, exiting gracefully")
            return
        
        cur_audio_file = files[index]
        cur_logit_file = cur_audio_file.replace('audio', 'logits')
        cur_audio_file = join(teacher_logits_dir, cur_audio_file)
        cur_logit_file = join(teacher_logits_dir, cur_logit_file)

        
        teacher_logits = torch.load(cur_logit_file, map_location='cpu', weights_only=True)
        audio_file = torch.load(cur_audio_file, map_location='cpu', weights_only=True)

        partial_load_asr_model_fns = []
        for cpt in config["checkpointing"]["asr_models"]:
            asr_model_checkpoint = torch.load(cpt, map_location="cpu", weights_only=False)
            asr_model_config = asr_model_checkpoint['config']
            asr_model_state_dict = asr_model_checkpoint['model']
            partial_load_asr_model_fns.append(partial(
                load_asr_model_fn,
                load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
                asr_model_state_dict,
            ))

      
  
        # policy_net = load_rl_models(config)
        # load_policy(policy_net, config
        augmentation = SpecAugment(n_time_masks=0, n_freq_masks=6, freq_mask_param=34, zero_masking=True, time_mask_param=0)
        

      

        rewards = []
        mask_list = []
        for i in range(config['repeats']):
            #with torch.no_grad(): policy_output = policy_net.augment(audio_file)
            audio_a = audio_file
            masks = augmentation(torch.ones_like(audio_file))
            audio_b = audio_file * masks + (1 - masks) * audio_file.mean().unsqueeze(0).unsqueeze(0)
            # masks = augmentation(torch.ones_like(audio_file))
            # audio_a = audio_file * masks + (1 - masks) * audio_file.mean().unsqueeze(0).unsqueeze(0)
  

            c_rewards = []
            for asr_model_load_fn in partial_load_asr_model_fns:
                reward, _ = cpu_rollout(
                    policy = None,
                    audio = audio_file,
                    audio_a = audio_a,
                    audio_b = audio_b,
                    load_asr_model_fn = asr_model_load_fn,
                    tokenizer = tokenizer,
                    teacher_logits = teacher_logits,
                    optim_args = {"lr":1e-2}
                )
                # rewards.append(reward)
                # mask_list.append(masks.to(dtype=torch.bool))
                c_rewards.append(reward)
            print('->', torch.stack(c_rewards).mean())
            rewards.append(torch.stack(c_rewards).mean())
            mask_list.append(masks.to(dtype=torch.bool))
            
            print('---')
            #print(reward)
        
        if config['save']:
            torch.save({
                'reward': torch.stack(rewards),
                'mask': torch.stack(mask_list)
            }, path)
          
            #for i, (reward, mask) in enumerate(zip(rewards, mask_list)):
                # while True:
                #     reward_file = files[index].replace('audio', 'reward').replace('.pt', f'_repeat_{i+offset}.pt')
                #     mask_file = files[index].replace('audio', 'masks').replace('.pt', f'_repeat_{i+offset}.pt')
                #     reward_file = join(save_path, reward_file)
                #     mask_file = join(save_path, mask_file)
                #     if os.path.exists(reward_file) or os.path.exists(mask_file):
                #         offset += 1
                #     else:
                #         break
            
                # torch.save(reward, reward_file)
                # torch.save(mask, mask_file)

    # torch.save(reward, join(save_path, reward_file))
    # torch.save(masks, join(save_path, mask_file))
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    parser.add_argument('--steps', '-steps', type=int, default=134)
    parser.add_argument('--repeats', '-repeats', type=int, default=100)
    parser.add_argument('--dont_save', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    config['steps'] = args.steps
    config['repeats'] = args.repeats
    config['save'] = not args.dont_save
    main(config)




