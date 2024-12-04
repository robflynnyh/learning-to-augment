import torch
from tqdm import tqdm
from l2augment.utils.data import prepare_chunks

def gen_logits(
        asr_model:torch.nn.Module,
        audio:torch.Tensor,
        seq_len:int=4096,
        overlap=0.875
    ):
    dtype = torch.float32 

    overlap = round(seq_len*overlap)
    audio = audio.to(dtype=dtype) #temporary
    audio_n = audio.shape[-1]
    asr_model = asr_model.to(dtype=dtype)

    if seq_len > audio_n:
        seq_len, overlap = audio_n, 0
   
    training_data, training_keys = prepare_chunks(audio, seq_len, overlap)
    training_keys = list(training_data.keys())

    model_outputs = []
    asr_model.eval()

    for i, key in tqdm(enumerate(training_keys)):
        audio_chunk = training_data[key].clone()
        with torch.no_grad():
            out = asr_model(audio_signal = audio_chunk)
        logits = out['final_posteriors'][0]
        logits = torch.exp(logits) # convert to prob
        ds_len = logits.shape[-2]
        ratio = audio_chunk.shape[-1] / ds_len
        overlap_ds = int(overlap / ratio)
        model_outputs[key] = {'logits': logits, 'ds_len': ds_len, 'overlap_ds': overlap_ds, 'index': i}

    return model_outputs

    

        
       









