

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