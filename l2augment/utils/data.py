import re
import os
from os.path import join
import json
from lcasr.utils.audio_tools import processing_chain, total_seconds, to_spectogram, total_frames
from lcasr.eval.utils import zero_out_spectogram
import pickle
from typing import List
import torch, torchaudio
from typing import Tuple
from torch.utils.data import Dataset
from l2augment.utils.helpers import lmap



class CustomDataset(Dataset):
    def __init__(
            self, 
            files, 
            zero_mean=True, 
            standardize_std=True, 
            divide_by_100=False,
            scale=False, 
            clamp_min=-5, 
            clamp_max=5,
            randomize_order=False, # debug
            decrease_measurement='absolute', # percentage or absolute
            load_audio=True,
            cer_weight=0.0,
            wer_weight=1.0,
            set_minus_or_positive=False,
            expand_mask_to_audio=False,
            all_zero_to_one=False,
            logger=print
        ):
        self.data = sorted(files)
    
        self.zero_mean = zero_mean
        self.standardize_std = standardize_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = scale
        self.randomize_order = randomize_order
        self.decrease_measurement = decrease_measurement
        self.divide_by_100 = divide_by_100
        self.load_audio = load_audio
        self.cer_weight = cer_weight
        self.wer_weight = wer_weight
        self.set_minus_or_positive = set_minus_or_positive
        self.expand_mask_to_audio = expand_mask_to_audio
        self.logger = logger
        self.all_zero_to_one = all_zero_to_one

    def __len__(self):
        # Return the total number of samples
        return len(self.data)
    
    def standardize_pipeline(self, rewards):
        
        if self.set_minus_or_positive:
            rewards[rewards < 0] = -1
            rewards[rewards > 0] = 1

        rewards_mean = rewards.mean(0, keepdim=True)

        if self.zero_mean:
            rewards = rewards - rewards_mean
        if self.divide_by_100:
            rewards = rewards / 100
            
        if self.clamp_min is not None:
            rewards = rewards.clamp(min=self.clamp_min)
        if self.clamp_max is not None:
            rewards = rewards.clamp(max=self.clamp_max)

        if self.standardize_std:
            rewards_std = rewards.std(0, keepdim=True)
            if rewards.shape[0] > 1 and rewards_std.sum() > 0:
                rewards = rewards / (rewards_std + 1e-6)
        if self.scale:
            # min -1, max 1 but avoid 0 division
            rewards_min = rewards.min(dim=0, keepdim=True).values
            rewards_max = rewards.max(dim=0, keepdim=True).values
            if rewards_min == rewards_max:
                rewards = torch.zeros_like(rewards) # best outcome
            else:
                rewards = 2*(rewards - rewards_min)/(rewards_max - rewards_min) - 1

        if self.all_zero_to_one:
            # if all rewards are zero then set them to 1 instead
            if rewards.min() == 0 and rewards.max() == 0:
                rewards = torch.ones_like(rewards)

        return rewards

    
    def __getitem__(self, idx):
        try:
            file = self.data[idx]
            rollout = torch.load(file, weights_only=True)

            audio = rollout['audio'] 
            masks = rollout['mask'] # kept in float8 for memory

            if self.expand_mask_to_audio: # when we have no time dim on mask and we want to expand it to the audio length
                masks = masks.unsqueeze(-1).repeat(1, 1, 1, audio.shape[-1])

            audio = audio if self.load_audio else None

            decreases = rollout['reward']   

            misc = {}
            if 'probs' in rollout:
                misc['probs'] = rollout['probs']
            if 'eps' in rollout:
                misc['eps'] = rollout['eps']
            if 'generation' in rollout:
                misc['generation'] = rollout['generation']


            if decreases.ndim == 3: has_wer = True
            else: has_wer = False   

            before, after = decreases.chunk(2, dim=-1)


            if self.decrease_measurement == 'absolute':
                rewards = before - after
            elif self.decrease_measurement == 'percentage':
                rewards = ( (before - after) / before )*100
            else:
                raise ValueError(f"Unknown decrease measurement: {self.decrease_measurement} should be 'percentage' or 'absolute'")
            rewards = rewards.squeeze(-1)

            # replace any nan values with 0 
            rewards[torch.isnan(rewards)] = 0 # can happen due to empty reference and hypothesis
            
            if has_wer:
                cer, wer = rewards.unbind(-1)
                total_reward = (cer*self.cer_weight + wer*self.wer_weight).sum()
                cer, wer = lmap(self.standardize_pipeline, [cer, wer])
                rewards = cer*self.cer_weight + wer*self.wer_weight
            else:
                total_reward = rewards.sum()
                rewards = self.standardize_pipeline(rewards)

            all_rewards = rewards

            return {
                'reward': all_rewards, # (repeats)
                'total_reward': total_reward,
                'masks':masks, # (masks, 1, C, T)       
                **({'audio':audio} if self.load_audio else {}),
                **misc
            }
        except Exception as e:
            self.logger(f"Error loading data: {e}")
            return None


def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def open_txt(path:str) -> str:
    with open(path, 'r') as f:
        return f.read().strip()

def convert_str_to_seconds(time_str:str) -> float: # in format: HH:MM:SS convert to seconds
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def segment_spectrogram(spec, frames_per_second, utterances):
    for utt in utterances:
        start,end = utt['start'], utt['end']
        start_frame = int(round(start * frames_per_second))
        end_frame = int(round(end * frames_per_second))
        utt['spectrogram'] = spec[:, :, start_frame:end_frame].clone()
    return utterances   

def trim_spec(spec:torch.Tensor, start_w:float, end_w:float): # spec: (B, F, T)
    '''Trim the spectrogram to the start of the first word and the end of the last word.'''
    start_frame, end_frame = list(map(total_frames, [start_w, end_w]))
    return spec[:, :, start_frame:end_frame]


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

    def load_tedlium_recording(stm_path:str, sph_path:str):
        utts = proc_stm_and_timings(stm_path)
        audio= processing_chain(sph_path, normalise=True) # [b, c, t]
        length_in_seconds = total_seconds(audio.shape[-1])
        frames_per_second = audio.shape[-1] / length_in_seconds
        utterances = segment_spectrogram(audio, frames_per_second, utts)
        return utterances

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

def tedlium3_data():
    default_base_path = "/mnt/parscratch/users/acp21rjf/TEDLIUM_release-3/legacy/"

    def proc_stm_and_timings(stm_path:str):
        stm = open_stm(stm_path)
        all_text = ""
        timings = []
        remove_timings = []
        for line in stm:
            sline = line.split(' ')
            if len(sline) < 6:
                continue
            a_id, s_id, spk, start, end, meta = sline[:6]
            text = ' '.join(sline[6:])
            if text == 'ignore_time_segment_in_scoring':
                remove_timings.append({'start': float(start), 'end': float(end)})
                continue
            all_text += text + ' '
            timings.append({'start': float(start), 'end': float(end)})
        all_text = all_text.strip()
        # regex to do all of the above
        # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
        all_text = re.sub(r" '([a-z])", r"'\1", all_text)
        # remove multiple spaces
        all_text = re.sub(r" +", r" ", all_text)
        return all_text, timings, remove_timings

    def fetch_data(path:str):
        audio_path = os.path.join(path, 'sph')
        audio_files = [os.path.join(audio_path, el) for el in os.listdir(audio_path) if el.endswith('.sph')]
        audio_files.sort()
        text_path = os.path.join(path, 'stm')
        text_files = [os.path.join(text_path, el) for el in os.listdir(text_path) if el.endswith('.stm')]
        text_files.sort()
        assert len(audio_files) == len(text_files), 'Number of audio files and text files must match'
        return audio_files, text_files

    def process_text_and_audio_fn(rec_dict, single_utterance=False):
        audio, text = rec_dict['audio'], rec_dict['text']
        audio_spec = processing_chain(audio)

        gold_text, _, remove_timings = proc_stm_and_timings(stm_path=text)
        audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
        return audio_spec, (gold_text).lower().strip()
       

    def get_text_and_audio(split, base_path=None, **kwargs):
        assert split in ['test', 'dev', 'train'], f'Split must be either test or dev train (got {split})'
        base_path = base_path or default_base_path
        path = os.path.join(base_path, split)
        
        audio_files, text_files = fetch_data(path=path)
        return_data = []
        for rec in range(len(audio_files)):
            return_data.append({
                'id': audio_files[rec],
                'text': text_files[rec], 
                'audio': audio_files[rec], 
                "process_fn": process_text_and_audio_fn
            })
        return_data = sorted(return_data, key=lambda x: x['id'])

        return return_data
    return get_text_and_audio


def rev16_data():
    default_base_path = '/mnt/parscratch/users/acp21rjf/rev_benchmark'
    #TEST_IDS = '/mnt/parscratch/users/acp21rjf/rev_benchmark/test.txt'

    def fetch_data(data_path:str, ids:str):
        with open(ids, 'r') as f:
            IDS = f.read().strip().split(" ")
            IDS = [el.strip() for el in IDS if el.strip() != '']

        audio_files = [{
            'id': el,
            'path': os.path.join(data_path, "audio", el+".mp3"),
        } for el in IDS]

        text_files = [{
            'id': el,
            'text': open_txt(os.path.join(data_path, "transcripts", el+".txt"))
        } for el in IDS]


        return audio_files, text_files

    def preprocess_transcript(text:str): return text.lower()
    def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])

    def get_text_and_audio(split, base_path=None, **kwargs):
        assert split in ['test'], 'Split must be test'
        data_path = base_path or default_base_path
        test_ids = os.path.join(data_path, 'test.txt')
        
        audio_files, text_files = fetch_data(data_path = data_path, ids = test_ids)
        return_data = []
        for rec in range(len(audio_files)):
            return_data.append({
                'text': text_files[rec]['text'], 
                'audio': audio_files[rec]['path'], 
                "process_fn": process_text_and_audio_fn,
                "id": text_files[rec]['id'],
            })
        return_data = sorted(return_data, key=lambda x: x['id'])
        return return_data
    
    return get_text_and_audio

def chime6_data():

    default_basedir = "/mnt/parscratch/users/acp21rjf/chime6/"
    TEST_AUDIO = 'audio/eval'
    DEV_AUDIO = 'audio/dev'
    TEST_TEXT = 'transcriptions/eval'
    DEV_TEXT = 'transcriptions/dev'
    
    def get_data_paths(base_dir:str = default_basedir):
        return {
            'test': {
                'audio': join(base_dir, TEST_AUDIO),
                'text': join(base_dir, TEST_TEXT)
            },
            'dev': {
                'audio': join(base_dir, DEV_AUDIO),
                'text': join(base_dir, DEV_TEXT)
            }
        }



    def combine_and_load_audio(audio_files:list, stime:float, etime:float) -> torch.Tensor:
        '''Here we take all the channels for the first microphone array (U01) and combine them via averaging the spectrograms and normalizing the result'''
        # load all audio files
        audios = []
        for audio_file in audio_files:
            audio, _ = torchaudio.load(audio_file)
            audio = audio.mean(dim=0)
            audios.append(audio)
        max_len = max([audio.shape[-1] for audio in audios])
        # pad from the right
        audios = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1]))[None] for audio in audios]
    
        specs = [to_spectogram(waveform=audio, global_normalisation=False) for audio in audios]
        # get duration in seconds    
        specs = [trim_spec(spec, stime, etime) for spec in specs]
        spec = torch.stack(specs, dim=0).mean(dim=0)
        # renormalize
        spec = (spec - spec.mean(-1, keepdim=True)) / spec.std(-1, keepdim=True)

        return spec

    def fetch_data(data:dict = get_data_paths()['test']) -> Tuple[list, list]:
        # get text
        text_files = {}
        start_times = {}
        end_times = {}
        for filename in os.listdir(data['text']):
            if filename.endswith('.json'):
                with open(os.path.join(data['text'], filename), 'r') as f:
                    j_data, Sname = json.load(f), filename.replace('.json', '')
                    text_files[Sname] =  " ".join([el['words'] for el in j_data])
                    stime, etime = list(map(convert_str_to_seconds, [j_data[0]['start_time'], j_data[-1]['end_time']]))
                    start_times[Sname] = stime
                    end_times[Sname] = etime
        
        # get audio
        audio_file_names = [el for el in os.listdir(data['audio']) if el.endswith('.wav')]
        #audio_file_names = [el for el in audio_file_names if re.match('S\d+_U01.CH1.wav',el)] #\d+.wav', el)]
        audio_file_names = [el for el in audio_file_names if re.match('S\d+_U01.CH\d+.wav', el)]

        # get unique scene names
        scene_names = list(set([el.split('_')[0] for el in audio_file_names]))
        audio_files = {k:[] for k in scene_names}
        for filename in audio_file_names:
            scene_name = filename.split('_')[0]
            audio_files[scene_name].append(os.path.join(data['audio'], filename))

        # check keys match for audio and text
        assert set(audio_files.keys()) == set(text_files.keys()), 'Keys do not match'
            
        return audio_files, text_files, start_times, end_times

    def process_text_and_audio_fn(rec_dict): 
        return combine_and_load_audio(rec_dict['audio'], rec_dict['stimes'], rec_dict['etimes']), rec_dict['text'].lower()

    def get_text_and_audio(split, base_path=None, **kwargs):
        assert split in ['test', 'dev'], f'Split must be either test or dev (got {split})'
        base_path = base_path or default_basedir
        data_path = get_data_paths(base_dir=base_path)[split]
        audio_files, text_files, stimes, etimes = fetch_data(data=data_path)
        return_data = []
        
        for rec in list(audio_files.keys()):
            return_data.append({
                'id': rec,
                'text': text_files[rec], 
                'audio': audio_files[rec], 
                'stimes': stimes[rec],
                'etimes': etimes[rec],
                "process_fn": process_text_and_audio_fn
            })

        return return_data

    
    return get_text_and_audio


dataset_functions = {
    "earnings22": earnings22_data(),
    "this_american_life": this_american_life_data(),
    "tedlium3_segmented_data": tedlium3_segmented_data(),
    "tedlium": tedlium3_data(),
    "rev16": rev16_data(),
    "chime6": chime6_data()
}


def save_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionary(path):
    with open(path, 'rb') as file:
        return pickle.load(file)