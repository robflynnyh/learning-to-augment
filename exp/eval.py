import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout.cpu_multistep     import  cpu_rollout
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


AUDIO_CHUNK_SIZE_DEFAULT = 2048
AUDIO_CHUNK_OVERLAP_DEFAULT = 0


def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    asr_model.flash_attn = False
    return asr_model

def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


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

def load_policy(model, config, path=None):
    save_path = config.get('training', {}).get('model_save_path', None) if path == None else path
    if save_path == None:
        return 
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

def main(config, policy_net=None):

    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    
    asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
    asr_model_config = asr_model_checkpoint['config']
    asr_model_state_dict = asr_model_checkpoint['model']

    partial_load_asr_model_fn = partial(
        load_asr_model_fn,
        load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class),
        asr_model_state_dict,
    )

    if policy_net is None:
        policy_net = load_rl_models(config)
        load_policy(policy_net, config)
   
    rollout_fn = partial(cpu_rollout, 
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
    
    

        rollout_output = rollout_fn(
            policy = policy_net,
            audio = audio_spec,
            text = gold_text,
            augmentation_config = config.get('evaluation', {}).get('augmentation_config', {}),
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
    if save_path != "":
        id = config.get('evaluation', {}).get('id', "0") 
        results = f"ID: {id} - Dataset: {dataset} - Split: {split} - Epochs: {epochs} - Original_WER: {original_wer} - Updated_WER: {wer}"
        print(results)
        raise NotImplementedError


    return wer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--indexes', '-indexes', type=int, nargs='+', help='Indexes of the data to evaluate', default=[-1]) # -1 means all
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['indexes'] = args.indexes
    main(config)




