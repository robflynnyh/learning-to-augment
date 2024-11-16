import torch
from madgrad import MADGRAD
from torch import Tensor
from typing import List, Dict, Any, Callable
from torch.nn import Module
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
import random
from functools import partial
from einops import repeat, rearrange

DEFAULT_OPTIMIZER_CLASS = MADGRAD
MAX_UTT_LIMIT = 15 #LIMIT TO 10 FOR NOW
CHUNK_OVERLAP = 0

def model_vmap_fn(model):
  def run_fn(weights, spectrogram):
    return torch.func.functional_call(module=model, parameter_and_buffer_dicts=[weights], args=spectrogram)
  return run_fn

def calc_loss_fn(model, spectrogram, weights, loss_fn):
  def run_fn(grad_weights):
    fwd_weights = {k:grad_weights[k] if k in grad_weights else v for k,v in weights.items()}
    output = torch.func.vmap(model_vmap_fn(model))(fwd_weights, spectrogram)
    noisy_predictions = output['final_posteriors'].squeeze(1)
    loss = loss_fn(noisy_predictions)
    return loss, output
  return run_fn

def apply_ctc_loss_fn(ctc_loss_fn, pseudo_targets, a_lengths, target_lengths, total_tokens_in_loss, noisy_predictions):
  return ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, a_lengths, target_lengths) / total_tokens_in_loss


def gpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        audio:Tensor,
        text:List[Dict[str,str]],
        chunk_audio_function:Callable,
        chunk_text_function:Callable,
        tokenizer,
        audio_lengths,
        optim_args:Dict[str, Any] = {"lr":3e-5},
        **kwargs
    ):

    dtype = torch.float32 #temporary
    audio = audio.to(dtype=dtype) #temporary

    audio_chunks = chunk_audio_function(spec = audio)
    text_chunks = [chunk_text_function(text = el, spectogram_length=audio.shape[-1]) for el in text]
    #TODO text_chunks = [chunk_text_function(text=el, spectogram_length=audio.shape[-1])
    utterances = len(audio_chunks)
    batch_size = audio_chunks[0].size(0)

    chunks, culm_lengths_audio, nans_in_a_row = [], torch.zeros_like(audio_lengths), 0

    ################################
    i = 0
    tracker = torch.arange(len(audio_chunks))[None].repeat(batch_size, 1)
    #print(tracker)
    for ix, el in enumerate(audio_chunks):

        reset_mask = ~(culm_lengths_audio > audio_lengths)
        # print(reset_mask.shape, tracker.shape, ix, tracker[reset_mask].shape, tracker[reset_mask][:, ix, None].shape)
        tracker[~reset_mask] -= tracker[~reset_mask][:, ix, None]
        culm_lengths_audio[~reset_mask] *= 0
        #print(reset_mask)

        cur_chunks = [audio_chunks[t_id[ix]][t_idx] for t_idx, t_id in enumerate(tracker)]
        cur_chunk_lens = [el.size(-1) for el in cur_chunks]
        if min(cur_chunk_lens) != max(cur_chunk_lens):
           for c_i in range(len(cur_chunks)):
              if cur_chunks[c_i].size(-1) < max(cur_chunk_lens):
                 cur_chunks[c_i] = torch.cat((cur_chunks[c_i], torch.zeros((cur_chunks[c_i].size(0), max(cur_chunk_lens)-cur_chunks[c_i].size(-1)))), dim=-1)
      
        #print([el.shape for el in cur_chunks])
        cur_chunks = torch.stack(cur_chunks, dim=0)

        cur_culm_lengths = culm_lengths_audio
        cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths - CHUNK_OVERLAP).clamp(0)
      
        txt_chunks = [el[tracker[i][ix]] for i, el in enumerate(text_chunks)]

        chunks.append({
            'audio':cur_chunks,
            'txt':txt_chunks,
            'audio_lengths':cur_lengths,
            'cur_culm_lengths':cur_culm_lengths,
        })
        culm_lengths_audio += cur_chunks.shape[-1] - (CHUNK_OVERLAP if ix != 0 else 0)
        i+=1
    ################################
    # for z,el in enumerate(chunks):
    #    print(z,[el['txt'][0]])

    random.shuffle(chunks)
    chunks = chunks[:MAX_UTT_LIMIT]
    
    asr_model = load_asr_model_fn()
    asr_model = asr_model.to('cuda', dtype=dtype)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='sum')
    verbose = kwargs.get("verbose", False)

    for chunk in chunks:
      text_sample = chunk['txt']
      with torch.no_grad():
        out = asr_model(audio_signal = chunk['audio'].to('cuda'), length=chunk['audio_lengths'].to('cuda'))
      initial_predictions = [decoder(out['final_posteriors'][i]) for i in range(out['final_posteriors'].size(0))]
      chunk['initial_wers'] = torch.tensor([word_error_rate_detail(hypotheses=[pred], references=[ref], use_cer=True)[0] for pred, ref in zip(initial_predictions, text_sample)])
      for i_wer in range(len(chunk['initial_wers'])):
         if chunk['initial_wers'][i_wer] == float('inf'): chunk['initial_wers'][i_wer] = 0.0
        
   
    asr_model.eval()
    asr_model_weights = {k:repeat(v, f'... -> {batch_size} ...').clone() for k,v in asr_model.state_dict().items()}
    optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model_weights.values(), **optim_args)
    

    rewards = torch.zeros((batch_size, len(chunks)))
    seeds = []
    masks = []

    for i, chunk in enumerate(chunks):
      audio_sample = chunk['audio'].to('cuda')
      audio_sample_lengths = chunk['audio_lengths'].to('cuda')
      text_sample = chunk['txt']
    
      with torch.no_grad(): policy_output = policy.augment(audio_sample, return_seed=True, return_mask=True)
      augmented_audio_sample = policy_output['augmented_data']
      seeds.append(policy_output['seed'].cpu())
      masks.append(policy_output['mask'].cpu())

      with torch.no_grad():
          model_fwd_func = torch.func.vmap(model_vmap_fn(asr_model))
          teacher_output = model_fwd_func(asr_model_weights, audio_sample.unsqueeze(1))
          teacher_output_length = teacher_output['length']
          teacher_output_posteriors = teacher_output['final_posteriors'].squeeze(1)
          
      pseudo_targets = [decoder(el) for el in teacher_output_posteriors]
      pseudo_targets = [torch.LongTensor(tokenizer.encode(el)) for el in pseudo_targets]
      pseudo_target_lengths = torch.LongTensor([el.size(0) for el in pseudo_targets]).to('cuda')
      enc_pseudo_targets = torch.nn.utils.rnn.pad_sequence(pseudo_targets, batch_first=True, padding_value=tokenizer.pad_id()).to('cuda')
      cur_tokens_in_loss = audio_sample_lengths.sum()

 
      partial_loss_fn = partial(apply_ctc_loss_fn, ctc_loss_fn, enc_pseudo_targets, teacher_output_length, pseudo_target_lengths, cur_tokens_in_loss)

      grad_and_output_func = torch.func.grad(calc_loss_fn(asr_model, augmented_audio_sample.unsqueeze(1), asr_model_weights, partial_loss_fn), has_aux=True)
      with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        grad_and_ouput = grad_and_output_func({k:v for k,v in asr_model_weights.items() if k.endswith('weight') or k.endswith('bias')})
      grad, output = grad_and_ouput[0], grad_and_ouput[1]
      
   
      optimizer.zero_grad()
      for k,v in grad.items():
         asr_model_weights[k].grad = v
      optimizer.step()
    
      with torch.no_grad():
        model_fwd_func = torch.func.vmap(model_vmap_fn(asr_model))
        updated_output = model_fwd_func(asr_model_weights, audio_sample.unsqueeze(1))
        updated_output_posteriors = updated_output['final_posteriors'].squeeze(1)

      updated_predictions = [decoder(el) for el in updated_output_posteriors]
      updated_wers = torch.tensor([word_error_rate_detail(hypotheses=[pred], references=[ref],use_cer=True)[0] for pred, ref in zip(updated_predictions, text_sample)])
      for i_wer in range(len(updated_wers)):
         if updated_wers[i_wer] == float('inf'): updated_wers[i_wer] = 0.0
      initial_wers = chunk['initial_wers']

      absolute_wer_reductions = initial_wers - updated_wers

      gamma_power = torch.arange(i+1).flip(0)[None].repeat(batch_size,1)
      reward_factor = 0.9**gamma_power
      reward_at_t = reward_factor*absolute_wer_reductions.unsqueeze(1)
    
      rewards[:, :reward_at_t.size(1)] += reward_at_t

    rewards = rewards
    seeds = torch.stack(seeds, dim=1)
    masks = torch.stack(masks, dim=1).squeeze(-1)

    return {
        'rewards': rewards,
        'masks': masks,
        'seeds': seeds
    }

 
        
       










