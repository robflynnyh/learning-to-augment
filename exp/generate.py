import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
from l2augment.utils.helpers import load_model as load_policy, load_asr_model_fn
from l2augment.rollout import  cpu_rollout
from l2augment.utils.helpers import load_rl_models
from l2augment.utils.data import dataset_functions

import os
from os.path import join
import json
import pickle
import random




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

def load_model(model, config):
    save_path = config.get('training',{}).get('model_save_path', None)
    if save_path == None: return
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

    optim_args = config['generation'].get('optim_args', {"lr": 1e-1})
    print(f"Optim Args: {optim_args}")
    augmentation_args = config['generation'].get('augmentation_config', {})
    print(f"Augmentation Args: {augmentation_args}")

    policy_path = config.get('training',{}).get('model_save_path', None)
    policy_net = load_rl_models(config)
    print(f'Loaded Policy Class: {policy_net.__class__.__name__}')
    if policy_path != None and os.path.exists(policy_path):
        load_policy(policy_net, config)


    for i in range(config['steps']):
        index = config['index']*config['steps'] + i


        save_path = config['generation']['save_dir']
        assert os.path.exists(save_path), f"Save path {save_path} does not exist"
        split = {'train':'train', 'val':'dev', 'dev':'dev'}[config['split']]
        save_path = os.path.join(save_path, split)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    
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
                                text = text,
                                optim_args = optim_args,
                                return_wer = config['generation'].get('return_wer', False),
            )      
            path = join(save_path, f'{utt_id}.pt')
            repeats = config['repeats']
          
            if config['save'] and not os.path.exists(path):
                if repeats == 1: repeats = 2 # to ensure that we can get mean and std stats on first run!
            if config['skip_percentage'] != 0.0 and random.random() * 100 <= config['skip_percentage']:
                print(f"Skipping {utt_id}")
                if config['remove_skipped_paths'] and os.path.exists(path):
                    os.remove(path)
                continue

            rewards = []
            mask_list = []
            other_outputs = {}
            for i in range(repeats):
                audio_a = audio.clone()
                outputs = policy_net.augment(audio_a, **augmentation_args)
                audio_b, noise = outputs[0], outputs[1]
                if len(outputs) > 2:
                    misc = outputs[2]
                    for k in misc:
                        if k not in other_outputs:
                            other_outputs[k] = []
                        if misc[k].dtype in [torch.float32]:
                            misc[k] = misc[k].to(torch.float16) # save space
                        other_outputs[k].append(misc[k])

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
                # if i == repeats - 1:
                mask_list.append(noise.to(dtype=torch.float8_e5m2))
                #print(reward)

            rewards = torch.stack(rewards)
            mask_list = torch.stack(mask_list)
            for k in other_outputs:
                other_outputs[k] = torch.stack(other_outputs[k])
        
            print(path)
         
            # if os.path.exists(path):
            #     print(torch.cat([torch.load(path)['reward'], torch.stack(rewards)]))
            # else:
            print(rewards)
            if config['save']:
                if os.path.exists(path) and config['buffer_size'] != 0:
                    prev_data = torch.load(path)

                    prev_reward = prev_data['reward']
                    prev_mask = prev_data['mask']
                    if config['buffer_size'] != -1:
                        prev_reward = prev_reward[-config['buffer_size']:]
                        prev_mask = prev_mask[-config['buffer_size']:]
                        for k in other_outputs:
                            if k in prev_data:
                                prev_data[k] = prev_data[k][-config['buffer_size']:]

                    for k in other_outputs:
                        if k in prev_data:
                            other_outputs[k] = torch.cat([prev_data[k], other_outputs[k]], dim=0)

                    
                    torch.save({
                        'reward': torch.cat([prev_reward, rewards]),
                        'mask': torch.cat([prev_mask, mask_list]),
                        'audio': audio.to(torch.float16),
                        **other_outputs
                    }, path)
                else:
                    torch.save({
                        'reward': rewards,
                        'mask': mask_list,
                        'audio': audio.to(torch.float16),
                        **other_outputs
                    }, path)
      
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    parser.add_argument('--steps', '-steps', type=int, default=5)
    parser.add_argument('--repeats', '-repeats', type=int, default=2)
    parser.add_argument('--split', '-split', type=str, default='train')
    parser.add_argument('--buffer_size', '-buffer_size', type=int, default=0)
    parser.add_argument('--dont_save', action='store_true')
    parser.add_argument('--skip_percentage', '-skip_percentage', type=float, default=0.0)
    parser.add_argument('--remove_skipped_paths', '-remove_skipped_paths', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    config['steps'] = args.steps
    config['repeats'] = args.repeats
    config['save'] = not args.dont_save
    config['split'] = args.split
    config['buffer_size'] = args.buffer_size
    config['skip_percentage'] = args.skip_percentage
    config['remove_skipped_paths'] = args.remove_skipped_paths
    main(config)




