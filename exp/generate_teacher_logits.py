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
from l2augment.utils.gen_teacher_logits import gen_logits
from l2augment.utils.data import dataset_functions, save_dictionary


import os
from os.path import join
import json
import pickle
import random

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0.875



def main(config):
    save_path = config['teacher_logits_generation']['save_dir']
    checkpoint = config['teacher_logits_generation']['asr_checkpoint']
    dataset = "this_american_life"
    
    r_id = f"{dataset}_{config['index']}.pt"
    save_path = join(save_path, r_id)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, exiting gracefully")
        return

    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    
    asr_model_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    asr_model_config = asr_model_checkpoint['config']
    asr_model_state_dict = asr_model_checkpoint['model']

    asr_model = load_asr_model(asr_model_config, tokenizer.vocab_size(), asr_model_class)
    asr_model.load_state_dict(asr_model_state_dict)
    asr_model.flash_attn = False

    data = dataset_functions[dataset]("train")

    try: cur_data = data[config['index']]
    except IndexError:
        print(f"Index {config['index']} out of range, exiting gracefully")
        return
    
    audio_spec, _ = cur_data['process_fn'](cur_data)

    logits = gen_logits(
        asr_model=asr_model,
        audio=audio_spec,
        seq_len=AUDIO_CHUNK_SIZE_DEFAULT,
        overlap=AUDIO_CHUNK_OVERLAP_DEFAULT,
        vocab_size=tokenizer.vocab_size() + 1,
    )

    torch.save(logits, save_path)
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--index', '-index', type=int, default=0)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['index'] = args.index
    main(config)




