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

import os
from os.path import join
import json
import pickle
import random

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0


EXT = '.mp3'
AUDIO_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/audio'
TRAIN_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/train-transcripts-aligned.json'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/valid-transcripts-aligned.json'
TEST_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/test-transcripts-aligned.json'

def fetch_data(txt_path:str):
    with open(txt_path, 'r') as f:
        txt_json = json.load(f)

    episodes = list(txt_json.keys())
    audio_files = [{'path':os.path.join(AUDIO_PATH, el.split('-')[-1] + EXT), 'id': el} for el in episodes]
    text = [{'id': el, 'text': " ".join([el2['utterance'] for el2 in txt_json[el]])} for el in episodes]
    speakers = [len(set([el2['speaker'] for el2 in txt_json[el]])) for el in episodes]

    return audio_files, text, speakers

def preprocess_transcript(text:str): return text.lower()

def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])


def get_text_and_audio(split):
    if split == 'train':
        data_path = TRAIN_PATH
    elif split == 'dev':
        data_path = DEV_PATH
    elif split == 'test':
        data_path = TEST_PATH
    elif split == 'all':
        return get_text_and_audio('train') + get_text_and_audio('dev') + get_text_and_audio('test')
    else:
        raise ValueError(f'Invalid split: {split}')
     
    audio_files, text, speakers = fetch_data(txt_path=data_path)
    return_data = []
    for rec in range(len(audio_files)):
        assert audio_files[rec]['id'] == text[rec]['id'], f'Episode names do not match: {audio_files[rec]["id"]}, {text[rec]["id"]}'
        return_data.append({
            'id': audio_files[rec]['id'],
            'text': text[rec]['text'], 
            'audio': audio_files[rec]['path'], 
            "process_fn": process_text_and_audio_fn,
            'speakers': speakers[rec]
        })
    return return_data

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
    save_path = config['generation']['save_dir']
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
    policy_net = load_rl_models(config)
    load_policy(policy_net, config)

    original_wer = find_existing_run_wer(directory=config['generation']['save_dir'], id=config['index'])
   
    rollout_fn = partial(cpu_rollout, 
                         load_asr_model_fn = partial_load_asr_model_fn, 
                         tokenizer = tokenizer, 
                         verbose = False, 
                         original_wer=original_wer,
                         max_steps = config['generation'].get('max_steps', None)
    )

    data = get_text_and_audio("train")
  
    cur_data = data[config['index']]
    audio_spec, gold_text = cur_data['process_fn'](cur_data)

    r_id = f"{config['index']}_{str(random.randint(0,99999999))}.pkl"

    rollout_output = rollout_fn(
        policy = policy_net,
        audio = audio_spec,
        text = gold_text,
    )

    print(rollout_output['original_wer'], rollout_output['updated_wer'])

    if save_path: # debug
        save_dictionary(
            rollout_output, 
            filename=join(save_path, r_id)
        )
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    main(config)




