import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout import cpu_rollout
from l2augment.modelling.models import Policy, Value
from lcasr.utils.audio_tools import load_json
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json, reset_seen_ids
import time
import logging
from tqdm import tqdm
import random
from typing import List
from os.path import join

AUDIO_CHUNK_SIZE_DEFAULT = 4096
AUDIO_CHUNK_OVERLAP_DEFAULT = 0
NUM_WORKERS_DEFAULT = 8
PIN_MEMORY_DEFAULT = False
PREFETCH_DEFAULT = 2
POLICY_OUTPUT_DIM_DEFAULT = 80
VALUE_OUTPUT_DIM_DEFAULT = 1


def load_rl_models(config): #TODO!
    policy_net = Policy(
        input_dim=config['policy']['input_dim'],
        output_dim=config['policy'].get('output_dim', POLICY_OUTPUT_DIM_DEFAULT)
    )
    value_net = Value(
        input_dim=config['value']['input_dim'],
        output_dim=config['policy'].get('output_dim', VALUE_OUTPUT_DIM_DEFAULT)
    )
    return policy_net, value_net

def load_asr_model_fn(asr_model, state_dict):
    asr_model.load_state_dict(state_dict)
    return asr_model

def get_random_seed(config):
    random_seed = config['training'].get('random_seed', 1234)
    if random_seed == 'random':  # generate using time
        random_seed = int(time.time()) % 10000 
        logging.info(f'random seed: {random_seed}')
    return random_seed
    
def add_base_path_to_paired_data(paired_data, base_path):
    for key in paired_data:
        paired_data[key]['audio'] = join(base_path, "audio", paired_data[key]['audio'])
        paired_data[key]['txt'] = join(base_path, "text", paired_data[key]['txt'])
    return paired_data

def load_paired_data(config):
    paired_data = load_json(config['data']['path'])
    paired_data = add_base_path_to_paired_data(paired_data=paired_data, base_path=config['data']['base'])
    return paired_data

def train_step(
        config,
        batch,
        rollout_fn,
        policy_net,
        value_net,
        tokenizer,
        seen_ids:List[str] = [],
        device='cuda',
    ):
        audio, audio_lengths, txt, ids = batch

        chunk_size = config["training"].get("audio_chunk_size", AUDIO_CHUNK_SIZE_DEFAULT)
        chunk_overlap = AUDIO_CHUNK_OVERLAP_DEFAULT

        chunk_audio_function = partial(chunk_spectogram, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        chunk_text_function = partial(chunk_text_json, chunk_size = chunk_size, chunk_overlap = chunk_overlap)

        cur_audio = audio[0,:,:audio_lengths[0]][None]
        cur_text = txt[0]

        policy_net.to('cpu')
        policy_net.eval()
        rollout_output = rollout_fn(
            policy = policy_net,
            audio = cur_audio,
            text = cur_text,
            chunk_audio_function = chunk_audio_function,
            chunk_text_function = chunk_text_function
        )
        rewards, masks, seeds = rollout_output['rewards'], rollout_output['masks'], rollout_output['seeds']

        rewards = rewards.to(device)
        masks = masks.to(device)
        seeds = seeds.to(device)
        policy_net.to(device)
        value_net.to(device)
        policy_net.train()
        value_net.train()

        predicted_rewards = value_net(masks).squeeze(-1)
  
        probs = policy_net(seed=seeds)
        log_prob_at_i = (masks*probs + (1-masks)*(1-probs)).log()
        prob_of_mask = torch.sum(log_prob_at_i, dim=-1)
      
        
    


def train_loop(
        config,
        dataloader,
        rollout_fn,
        policy_net,
        value_net,
        epoch:int=0,
        seen_ids:List[str] = []   
    ):
    max_epochs = config.get("training", {}).get("max_epochs", 1)
    i, finished = -1, False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings() * max_epochs
    pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')

    while not finished:
        try:
            batch, i = next(dataloader_iter), i + 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            epoch += 1
            seen_ids = reset_seen_ids(seen_ids = seen_ids, epoch = epoch - 1)
            if epoch >= max_epochs:
                finished = True
            else:
                dataloader.update(
                    batch_size = dataloader.batch_size, 
                    seen_ids = seen_ids,
                    random_seed = random.randint(0, 10000),
                )
                dataloader_iter = iter(dataloader)
                pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
            continue
        train_step(config, batch, rollout_fn, policy_net, value_net, dataloader.tokenizer, seen_ids=seen_ids)
    

    # for epoch in range(epochs):
    #     for batch in dataloader:
    #         utts, refs = batch["utts"], batch["refs"]
    #         rollout_fn(policy = policy_net)



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
    policy_net, value_net = load_rl_models(config)
    random_seed = get_random_seed(config=config)
    paired_data = load_paired_data(config=config)

    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = config['training']['batch_size'],
        chunk_size = config['training'].get('audio_chunk_size', AUDIO_CHUNK_SIZE_DEFAULT),
        chunk_overlap = 0,
        num_workers = config['training'].get('num_workers', NUM_WORKERS_DEFAULT),
        pin_memory = config['training'].get('pin_memory', PIN_MEMORY_DEFAULT),
        prefetch = config['training'].get('prefetch_factor', PREFETCH_DEFAULT),
        seen_ids = [], #TODO
        random_seed = random_seed,
    )
    rollout_fn = partial(cpu_rollout, load_asr_model_fn = partial_load_asr_model_fn, tokenizer = tokenizer, verbose = False)
    train_loop(
        config=config,
        dataloader=dataloader,
        rollout_fn=rollout_fn,
        policy_net=policy_net,
        value_net=value_net
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




