import re
import os
import json
from lcasr.utils.audio_tools import processing_chain
import pickle

def prepare_chunks(spec, seq_len, overlap):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())


def this_american_life_data():
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

    def process_text_and_audio_fn(rec_dict, frame_offset=0, num_frames=-1): 
        return processing_chain(rec_dict['audio'], frame_offset=frame_offset, num_frames=num_frames), preprocess_transcript(rec_dict['text'])


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
    return get_text_and_audio

def earnings22_data():
    TEST_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/test_original'
    DEV_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/dev_original'
    ALL_TEXT_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/full_transcripts.json'
    def fetch_data(audio_path:str = TEST_PATH, txt_path:str = ALL_TEXT_PATH):
        with open(txt_path, 'r') as f:
            all_text_json = json.load(f)

        audio_files = [{
            'meeting': el.replace('.mp3', ''),
            'path': os.path.join(audio_path, el)
            } for el in os.listdir(audio_path) if el.endswith('.mp3')]

        text_files = [{
            'meeting': el['meeting'],
            'text': all_text_json[el['meeting']]
            } for el in audio_files]
    
        return audio_files, text_files

    def preprocess_transcript(text:str):
        text = text.lower()
        text = text.replace('<silence>', '')
        text = text.replace('<inaudible>', '')
        text = text.replace('<laugh>', '')
        text = text.replace('<noise>', '')
        text = text.replace('<affirmative>', '')
        text = text.replace('<crosstalk>', '')    
        text = text.replace('…', '')
        text = text.replace(',', '')
        text = text.replace('-', ' ')
        text = text.replace('.', '')
        text = text.replace('?', '')
        text = re.sub(' +', ' ', text)
        return text

    def process_text_and_audio_fn(rec_dict, frame_offset=0, num_frames=-1): 
        return processing_chain(rec_dict['audio'], frame_offset=frame_offset, num_frames=num_frames), preprocess_transcript(rec_dict['text'])

    def get_text_and_audio(split):
        assert split in ['test', 'dev'], f'Split must be either test or dev (got {split})'
        data_path = TEST_PATH if split == 'test' else DEV_PATH
        audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
        return_data = []
        for rec in range(len(audio_files)):
            return_data.append({
                'id': audio_files[rec]['meeting'],
                'text': text_files[rec]['text'], 
                'audio': audio_files[rec]['path'], 
                "process_fn": process_text_and_audio_fn
            })
        return return_data
    return get_text_and_audio

dataset_functions = {
    "earnings22": earnings22_data(),
    "this_american_life": this_american_life_data()
}


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)