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
import json

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0
POLICY_OUTPUT_DIM_DEFAULT = 80

EXT = '.mp3'
AUDIO_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/audio'
TRAIN_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/train-transcripts-aligned.json'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/valid-transcripts-aligned.json'
TEST_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/test-transcripts-aligned.json'

def load_rl_models(config): 
    policy_net = Policy(
        input_dim=config['policy']['input_dim'],
        output_dim=config['policy'].get('output_dim', POLICY_OUTPUT_DIM_DEFAULT)
    )
    policy_net = policy_net
    return policy_net

def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    asr_model.flash_attn = False
    return asr_model

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

def main(config):
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
   
    rollout_fn = partial(cpu_rollout, load_asr_model_fn = partial_load_asr_model_fn, tokenizer = tokenizer, verbose = False)

    data = get_text_and_audio("train")


    # all_transcripts = load_json("/store/store4/data/earnings-22/full_transcripts.json")
    # file="/store/store4/data/earnings-22/train/4453085.spec.pt"
    # id = "4453085"
    # audio=torch.load(file, weights_only=True)
    # trainscript = all_transcripts[id]
    # print(audio.shape)
    
    # chunk_size = config["training"].get("audio_chunk_size", AUDIO_CHUNK_SIZE_DEFAULT)

    # rollout_output = rollout_fn(
    #     policy = policy_net,
    #     audio = audio,
    #     text = trainscript,
    # )
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




