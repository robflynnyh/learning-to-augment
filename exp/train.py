import argparse
import torch
from functools import partial
from omegaconf.omegaconf import OmegaConf
from lcasr.utils.audio_tools import load_tokenizer
from lcasr.utils.general import load_model as load_asr_model, get_model_class
# from l2augment.modelling import load_model as load_rl_models
from l2augment.rollout import cpu_rollout
from lcasr.utils.audio_tools import load_json
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json, reset_seen_ids
import time
import logging
from tqdm import tqdm
import random

def load_rl_models(*args, **kwargs): #TODO!
    return None, None

def load_asr_model_fn(config, vocab_size, model_class, state_dict):
    asr_model = load_asr_model(config, vocab_size, model_class)
    asr_model.load_state_dict(state_dict)
    return asr_model

def get_random_seed(config):
    random_seed = config['training'].get('random_seed', 1234)
    if random_seed == 'random':  # generate using time
        random_seed = int(time.time()) % 10000 
        logging.info(f'random seed: {random_seed}')
    return random_seed
    
def train_step(
        config,
        batch,
        rollout_fn,
        policy_net,
        value_net,
        tokenizer
    ):
        audio, audio_lengths, txt, ids = batch

        chunk_size = config["training"].get("audio_chunk_size", 4096)
        chunk_overlap = 0
        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        txt_chunks = [chunk_text_json(text = el, chunk_size = chunk_size, chunk_overlap = chunk_overlap, spectogram_length = audio.shape[-1]) for el in txt] 

        del audio
        backwards_every_loss, steps_since_backwards = 0.0, 0
        chunks, culm_lengths_audio, nans_in_a_row = [], torch.zeros_like(audio_lengths), 0
        pad_id = tokenizer.pad_id()

        for ix, el in enumerate(audio_chunks_):

            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - chunk_overlap).clamp(0)
          
            enc_txt_chunks = [torch.LongTensor(tokenizer.encode(el[ix])) for i, el in enumerate(txt_chunks) if remove_mask[i]]
            enc_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in enc_txt_chunks])
            enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(enc_txt_chunks, batch_first=True, padding_value=pad_id)
            if enc_txt_chunks_lengths.max() == 0:
                continue # skip if none contain text (bad batch)
            chunks.append({
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
            })
            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)


def train_loop(
        config,
        dataloader,
        rollout_fn,
        policy_net,
        value_net   
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
        train_step(config, batch, rollout_fn, policy_net, value_net, dataloader.tokenizer)
    

    # for epoch in range(epochs):
    #     for batch in dataloader:
    #         utts, refs = batch["utts"], batch["refs"]
    #         rollout_fn(policy = policy_net)



def main(config):
    tokenizer = load_tokenizer()
    asr_model_class = get_model_class(config = config)
    
    asr_model_state_dict = torch.load(config["checkpointing"]["asr_model"], map_location="cpu")["model"]
    partial_load_asr_model_fn = partial(
        load_asr_model_fn, 
        config, 
        tokenizer.vocab_size(), 
        asr_model_class, 
        asr_model_state_dict
    )
    policy_net, value_net = load_rl_models(config)

    random_seed = get_random_seed(config=config)

    paired_data = load_json(args.config['data']['path'])
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.config['training']['batch_size'],
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
        prefetch = args.prefetch_factor,
        seen_ids = [], #TODO
        random_seed = random_seed,
    )


    rollout_fn = partial(cpu_rollout, load_asr_model_fn = partial_load_asr_model_fn, tokenizer = tokenizer, verbose = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)




