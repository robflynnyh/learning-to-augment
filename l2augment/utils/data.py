import re
import os
import json
from lcasr.utils.audio_tools import processing_chain, total_seconds
import pickle
from typing import List

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    utts = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        text = re.sub(r" '([a-z])", r"'\1", text)
        # remove anything inside angle brackets i.e <...> 
        utts.append({'start': float(start), 'end': float(end), 'text': re.sub(r'<[^>]*>', '', text)})
    return utts

def segment_spectrogram(spec, frames_per_second, utterances):
    for utt in utterances:
        start,end = utt['start'], utt['end']
        start_frame = int(round(start * frames_per_second))
        end_frame = int(round(end * frames_per_second))
        utt['spectrogram'] = spec[:, :, start_frame:end_frame].clone()
    return utterances

def load_tedlium_recording(stm_path:str, sph_path:str):
    utts = proc_stm_and_timings(stm_path)
    audio= processing_chain(sph_path, normalise=True) # [b, c, t]
    length_in_seconds = total_seconds(audio.shape[-1])
    frames_per_second = audio.shape[-1] / length_in_seconds
    utterances = segment_spectrogram(audio, frames_per_second, utts)
    return utterances

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

def tedlium3_segmented_data():
    default_base_path = "/mnt/parscratch/users/acp21rjf/TEDLIUM_release-3/legacy/"

    def process_text_and_audio_fn(rec_dict):
        utterances = load_tedlium_recording(stm_path=rec_dict['text'], sph_path=rec_dict['audio'])
        return utterances

    def get_text_and_audio(split, base_path=None):
        base_path = base_path or default_base_path
        assert split in ['test', 'dev', 'train'], f'Split must be either test or dev or train(got {split})'
        path = os.path.join(base_path, split)
        recordings = os.listdir(os.path.join(path, 'sph'))

        return_data = []
        for rec in range(len(recordings)):
            return_data.append({
                'id': recordings[rec].replace('.sph', ''),
                'text': os.path.join(path, 'stm', recordings[rec].replace('.sph', '.stm')),
                'audio': os.path.join(path, 'sph', recordings[rec]),
                "process_fn": process_text_and_audio_fn
            })
        return return_data
    return get_text_and_audio

dataset_functions = {
    "earnings22": earnings22_data(),
    "this_american_life": this_american_life_data(),
    "tedlium3_segmented_data": tedlium3_segmented_data()
}


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)