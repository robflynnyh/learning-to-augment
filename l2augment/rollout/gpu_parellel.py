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

DEFAULT_OPTIMIZER_CLASS = torch.optim.SGD

def model_vmap_fn(model):
  def run_fn(weights, spectrogram):
    return torch.func.functional_call(module=model, parameter_and_buffer_dicts=[weights], args=spectrogram)
  return run_fn

def calc_loss_fn(model, spectrogram, weights, loss_fn, divide_by_lengths):
  def run_fn(grad_weights):
    fwd_weights = {k:grad_weights[k] if k in grad_weights else v for k,v in weights.items()}
    output = torch.func.vmap(model_vmap_fn(model))(fwd_weights, spectrogram)
    noisy_predictions = output['final_posteriors'].squeeze(1)
    loss = loss_fn(noisy_predictions)
    loss = loss / divide_by_lengths
    loss = loss.sum()
    
    return loss, output
  return run_fn

def apply_ctc_loss_fn(ctc_loss_fn, pseudo_targets, a_lengths, target_lengths, noisy_predictions):
  return ctc_loss_fn(noisy_predictions.transpose(0, 1), pseudo_targets, a_lengths, target_lengths) 


def broadcast_multiply(tensor1, tensor2):
    """
    Multiply a tensor of size B with a tensor of size (B, ...) 
    where the remaining dimensions can vary.
    
    Args:
    - tensor1 (torch.Tensor): Tensor of size B
    - tensor2 (torch.Tensor): Tensor of size (B, ...)
    
    Returns:
    torch.Tensor: Result of broadcasting multiplication
    """
    # Reshape tensor1 to have additional singleton dimensions for broadcasting
    # This ensures tensor1 can be multiplied with tensor2
    broadcast_tensor1 = tensor1.view(tensor1.size(0), *([1] * (tensor2.ndim - 1)))
    
    # Multiply with broadcasting
    return broadcast_tensor1 * tensor2

def gpu_rollout(
        policy:Module,
        load_asr_model_fn:Callable,
        tokenizer,
        audio_a:Tensor, # clean
        audio_b:Tensor, # clean + policy(clean)
        text:List[str],
        device:torch.device,
        audio_lengths:Tensor=None,
        dtype:torch.dtype = torch.float32,
        optim_args:Dict[str, Any] = {"lr":2e-1},
        **kwargs
    ):
    
    asr_model = load_asr_model_fn()
    asr_model = asr_model.to(device, dtype=dtype)
    asr_model.eval()
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = asr_model.decoder.num_classes-1)

    ctc_loss_fn = torch.nn.CTCLoss(blank=asr_model.decoder.num_classes-1, reduction='none')

    with torch.no_grad():
      out = asr_model(audio_signal = audio_a, length=None) # can't create mask inside vmap so not using lengths, can look at creating mask outside vmap
    out_lengths = out['length']
    initial_predictions = [decoder(out['final_posteriors'][i]) for i in range(out['final_posteriors'].size(0))]
    
    initial_cers = torch.tensor([word_error_rate_detail(hypotheses=[pred], references=[ref], use_cer=True)[0] for pred, ref in zip(initial_predictions, text)])
      
    batch_size = audio_a.size(0)
    asr_model.eval()
    asr_model_weights = {k:repeat(v, f'... -> {batch_size} ...').clone() for k,v in asr_model.state_dict().items()}
    #optimizer = kwargs.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)(asr_model_weights.values(), **optim_args)
    
    pseudo_targets = [torch.LongTensor(tokenizer.encode(el)) for el in initial_predictions]
    pseudo_target_lengths = torch.LongTensor([el.size(0) for el in pseudo_targets]).to(device)
    enc_pseudo_targets = torch.nn.utils.rnn.pad_sequence(pseudo_targets, batch_first=True, padding_value=tokenizer.pad_id()).to(device)
   

    partial_loss_fn = partial(apply_ctc_loss_fn, ctc_loss_fn, enc_pseudo_targets, out_lengths, pseudo_target_lengths)
    grad_and_output_func = torch.func.grad(calc_loss_fn(asr_model, audio_b.unsqueeze(1), asr_model_weights, partial_loss_fn, out_lengths), has_aux=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
      grad_and_ouput = grad_and_output_func({k:v for k,v in asr_model_weights.items() if k.endswith('weight') or k.endswith('bias')})
    grad, output = grad_and_ouput[0], grad_and_ouput[1]


    #optimizer.zero_grad()
    for k,v in grad.items():
        #asr_model_weights[k] = asr_model_weights[k] - v*optim_args['lr']  
        asr_model_weights[k] = asr_model_weights[k] - v*optim_args['lr']
    #optimizer.step()

    with torch.no_grad():
      model_fwd_func = torch.func.vmap(model_vmap_fn(asr_model))
      updated_output = model_fwd_func(asr_model_weights, audio_a.unsqueeze(1))
      updated_output_posteriors = updated_output['final_posteriors'].squeeze(1)

    updated_predictions = [decoder(el) for el in updated_output_posteriors]
    updated_cers = torch.tensor([word_error_rate_detail(hypotheses=[pred], references=[ref], use_cer=True)[0] for pred, ref in zip(updated_predictions, text)])

    diff = initial_cers - updated_cers
    return diff
        
       









