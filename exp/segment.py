import argparse
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm
from l2augment.utils.data import dataset_functions
import torch
from os.path import join


def main(config):
    split = {'train':'train', 'val':'dev', 'dev':'dev'}[config['split']]
    save_path = config['audio_samples'][split]
    data = dataset_functions['tedlium3_segmented_data'](config['split'], base_path=config.get('tedlium_path', None))

    for index in tqdm(range(len(data))):
        id = data[index]['id']
        utterances = data[index]['process_fn'](data[index])
        for utt_idx, utterance in enumerate(utterances):
            text = utterance['text']
            audio = utterance['spectrogram']
            utt_id = f'{id}_{utt_idx}'
            path = join(save_path, f'{utt_id}.pt')
            torch.save({
                'text': text,
                'audio': audio,
                'id': utt_id,
                'start': utterance['start'],
                'end': utterance['end'],
            }, path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--split', '-split', type=str, default='train')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config['split'] = args.split
    main(config)




