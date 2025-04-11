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
from l2augment.rollout.cpu_multistep_oracle import cpu_rollout_search

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

    optim_args = config['generation'].get('optim_args', {"lr":8e-6, 'single_step_lr': 9e-2})
    search_repeats = config['generation'].get('search_repeats', 5)
    rollout_repeats = config['generation'].get('rollout_repeats', 5)
    random_path = config['generation'].get('random_path', False)

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

        search_shuffle_seed = random.randint(999, 9999999)

        save_datas = {}
        for z in range(rollout_repeats):
            out = cpu_rollout_search(
                policy = policy_net,
                load_asr_model_fn = partial_load_asr_model_fn,
                tokenizer = tokenizer,
                utterances = utterances,
                asr_model_config = asr_model_config,
                asr_model_class = asr_model_class,
                optim_args = optim_args,
                shuffle=True,
                augmentation_config = augmentation_args,
                epochs = 1,
                search_repeats = search_repeats,
                shuffle_seed = search_shuffle_seed,
                random_path=random_path,
            )

            save_data = {}

            save_data['audio'] = out['audio']
            if 'generations' in out:
                save_data['generations'] = out['generations']
            if 'masks' in out and not config['dont_save_masks']:
                save_data['masks'] = out['masks']
            save_data['rewards'] = out['rewards']
            save_data['original_wer'] = out['original_wer']
            save_data['updated_wer'] = out['updated_wer']
            save_data['wer_decrease'] = out['original_wer'] - out['updated_wer']
            save_data['id'] = id
            save_data['top_idxs'] = out['top_idxs']
            save_data['rollout_repeat'] = z
            
            if 'n_losses' in out:
                save_data['n_losses'] = out['n_losses']
            if 'entropy' in out:
                save_data['entropy'] = out['entropy']

            path = join(save_path, f'{id}_{z}.pt')
            save_data['path'] = path
            save_datas[z] = save_data

        all_decreases = [save_datas[z]['wer_decrease'] for z in range(rollout_repeats)]
       
        if config['save']:
            for z in range(rollout_repeats):
                save_datas[z]['all_decreases'] = all_decreases
                
                torch.save(
                    save_datas[z],
                    save_datas[z]['path'],
                )
      
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    parser.add_argument('--steps', '-steps', type=int, default=5)
    parser.add_argument('--split', '-split', type=str, default='train')
    parser.add_argument('--buffer_size', '-buffer_size', type=int, default=0)
    parser.add_argument('--dont_save_masks', '-dont_save_masks', action='store_true')
    parser.add_argument('--dont_save', action='store_true')
    parser.add_argument('--skip_percentage', '-skip_percentage', type=float, default=0.0)
    parser.add_argument('--remove_skipped_paths', '-remove_skipped_paths', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    config['steps'] = args.steps
    config['save'] = not args.dont_save
    config['split'] = args.split
    config['buffer_size'] = args.buffer_size
    config['skip_percentage'] = args.skip_percentage
    config['dont_save_masks'] = args.dont_save_masks
    config['remove_skipped_paths'] = args.remove_skipped_paths
    main(config)




