import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.utils.helpers import load_model as load_policy, load_asr_model_fn

from l2augment.rollout.cpu_multistep_oracle import cpu_rollout as multistep_oracle_rollout

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


def main(config, policy_net=None):

    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    
    asr_model_checkpoint = torch.load(config["checkpointing"]["asr_model"], map_location="cpu", weights_only=False)
    asr_model_config = asr_model_checkpoint['config']
    asr_model_state_dict = asr_model_checkpoint['model']

    rollout_function = multistep_oracle_rollout

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
    

    dataset = 'tedlium3_segmented_data'
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
        utterances = cur_data['process_fn'](cur_data)
    
    

        rollout_output = rollout_fn(
            policy = policy_net,
            utterances = utterances,
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




