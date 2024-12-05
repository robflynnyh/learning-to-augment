import torch
from tqdm import tqdm
from l2augment.utils.data import prepare_chunks

def gen_logits(
        asr_model:torch.nn.Module,
        audio:torch.Tensor,
        vocab_size:int,
        seq_len:int=4096,
        overlap=0.875,
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

    model_outputs = {}
    asr_model.eval()

    all_logits, logit_count = torch.zeros((1, audio_n//4 + seq_len, vocab_size)), torch.zeros((1, audio_n//4 + seq_len, vocab_size))

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

    logit_position = 0
    for i in sorted(list(model_outputs.keys())):
        logits, ds_len, overlap_ds = model_outputs[i]['logits'], model_outputs[i]['ds_len'], model_outputs[i]['overlap_ds']
        logit_position -= overlap_ds if i != 0 else 0
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len    

    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    logits = torch.log(logits) 

    return logits 

    

        
       









