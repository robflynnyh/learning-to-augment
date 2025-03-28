from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange, repeat
from lcasr.components.batchrenorm import BatchRenorm1d
from torch.distributions import Normal
import random
from typing import Tuple, Callable, Dict
from einops.layers.torch import Rearrange
from lcasr.utils.augmentation import SpecAugment 
from contextlib import nullcontext
import matplotlib.pyplot as plt
import wandb

policy_dict = {}

class base(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

class SwiGlu(base):
    def __init__(
        self,
        input_dim:int,
        output_dim:int=None,
        expansion_factor:int=2,
        ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim

        self.in_layer = nn.Linear(input_dim, input_dim*expansion_factor*2)
        self.out_layer = nn.Linear(input_dim*expansion_factor, output_dim)
        self.act = nn.SiLU()
        
    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=-1)
        c = a * self.act(b)
        return self.out_layer(c)
    

class GatedConv1d(base):
    def __init__(
        self,
        input_dim:int,
        output_dim:int=None,
        expansion_factor:int=1,
        kernel_size:Tuple[int]=(1,1),
        stride:Tuple[int]=(1,1),
        padding:Tuple[int]=(0,0),
        dropout:float=0.0
        ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim

        self.in_layer = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*expansion_factor*2, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.out_layer = nn.Conv1d(in_channels=input_dim*expansion_factor, out_channels=output_dim, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=1)
        c = a * torch.nn.functional.silu(b)
        return self.out_layer(self.dropout(c))
    



class ResidualBlock(base):
    def __init__(
            self,
            module:nn.Module,
            ) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)

class DepthWiseSeparableConv1d(base):
    def __init__(
            self,
            input_dim:int,
            output_dim:int=None,
            kernel_size:int=1,
            stride:int=1,
            padding:int=0,
            ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim
        self.depthwise = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_dim)
        self.pointwise = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
    

    def forward(self, x:Tensor):
        return self.pointwise(self.depthwise(x))


class Policy(base):
    def __init__(self) -> None:
        super().__init__()
        if type(self) == Policy: raise Exception('Policy class is not meant to be instantiated directly. Use a subclass instead!')

    def augment(self, audio, *args, **kwargs):
        raise NotImplementedError
    
class NoAugmentationPolicy(Policy):
    def augment(self, audio, *args, **kwargs):
        return audio, torch.zeros_like(audio)

class FrequencyMaskingRanker(Policy):
    def __init__(self, zero_masking=True, epochs_until_random=-1) -> None:
        super().__init__()
        self.masker = SpecAugment(n_time_masks=0, n_freq_masks=6, freq_mask_param=34, zero_masking=True)
        self.zero_masking = zero_masking
        self.epochs_until_random = epochs_until_random


    def apply_mask(self, audio, mask):
        if not self.zero_masking:
            audio = audio * mask + (1 - mask) * audio.mean(dim=(1,2), keepdim=True)
        else:
            audio = audio * mask
        return audio
        
    def augment(self, audio, use_random=False, repeats=1, *args, **kwargs):
        assert audio.dim() == 3, 'audio must be 3D tensor'
        
        epoch = kwargs.get('epoch', 0)
        if self.epochs_until_random != -1 and epoch >= self.epochs_until_random: use_random = True

        if use_random:
            mask_spec = self.masker(torch.ones_like(audio))
            
            audio = self.apply_mask(audio, mask_spec)
            
            assert mask_spec.shape[1] == 80, 'mask_spec must have 80 channels'

            return audio, mask_spec[:,:,0] # mask is same across all time steps so we can just return the first one
        else: 
            return self.learnt_augmentation(audio, repeats=repeats)
        
    def learnt_augmentation(self, audio, repeats=1):
        raise NotImplementedError # must be implemented in subclass

class MixedMaskingRanker(Policy):
    def __init__(self, zero_masking=True, epochs_until_random=-1) -> None:
        super().__init__()
        self.zero_masking = zero_masking
        self.epochs_until_random = epochs_until_random

    @staticmethod
    def get_mask(audio):
        x = torch.ones_like(audio)
        min_p = random.random()/2
        time_masker = SpecAugment(n_time_masks=12, n_freq_masks=0, freq_mask_param=0, zero_masking=True, min_p=min_p)
        freq_masks = random.randint(5,7)
        freq_mask_param = random.randint(24, 44)
        freq_masker = SpecAugment(n_time_masks=0, n_freq_masks=freq_masks, freq_mask_param=freq_mask_param, zero_masking=True)
        method = random.randint(0,2)
        if method == 0:
            x = time_masker(x)
        elif method == 1:
            x = freq_masker(x)
        else:
            x = time_masker(x)
            x = freq_masker(x)
        return x

    def apply_mask(self, audio, mask):
        if not self.zero_masking:
            audio = audio * mask + (1 - mask) * audio.mean(dim=(1,2), keepdim=True)
        else:
            audio = audio * mask
        return audio
        
    def augment(self, audio, use_random=False, repeats=1, *args, **kwargs):
        assert audio.dim() == 3, 'audio must be 3D tensor'
        
        epoch = kwargs.get('epoch', 0)
        if self.epochs_until_random != -1 and epoch >= self.epochs_until_random: use_random = True

        if use_random:
            mask_spec = self.get_mask(audio)
            
            audio = self.apply_mask(audio, mask_spec)
            
            assert mask_spec.shape[1] == 80, 'mask_spec must have 80 channels'

            return audio, mask_spec
        else: 
            return self.learnt_augmentation(audio, repeats=repeats)
        
    def learnt_augmentation(self, audio, repeats=1):
        raise NotImplementedError # must be implemented in subclass


        
class TrainableFrequencyMaskingRanker(FrequencyMaskingRanker):
    def forward_pass(self, batch, device, **kwargs):
        network_inputs = {}
        network_inputs['mask'] = batch['masks'].to(device, dtype=torch.float32).squeeze(1) # B, 1, C -> B, C
        
        rewards = batch['rewards'].to(device)

        if 'audio' in batch: network_inputs['audio'] = batch['audio'].to(device)
        if 'lengths' in batch: network_inputs['lengths'] = batch['lengths'].to(device)
        if 'counts' in batch: network_inputs['counts'] = batch['counts'].to(device)

        score = self(**network_inputs)

        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(score, rewards, reduction='mean')
        elif self.loss_type == 'mult':
            loss = -torch.mean(score * rewards)
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        
        return loss, {'loss':loss}

class UnconditionalFrequencyMaskingRanker(TrainableFrequencyMaskingRanker):
    def __init__(self, zero_masking=True, loss_type='mse', epochs_until_random=-1) -> None:
        super().__init__(zero_masking, epochs_until_random)
        assert loss_type in ['mse', 'mult']
        self.loss_type = loss_type
        self.network = nn.Sequential(
            SwiGlu(input_dim=80, output_dim=1, expansion_factor=3),
            Rearrange('b 1 -> b') 
        )


    def learnt_augmentation(self, audio, repeats=1):
        b, c, t = audio.shape
        assert c == 80, 'audio must have 80 channels'
        masks = torch.ones(b * repeats, c, 1).to(audio.device)
        masks = self.masker(masks).squeeze(-1)
        mask_scores = self(masks)
        masks = rearrange(masks, '(b r) c -> b r c', r=repeats)
        masks_scores = rearrange(mask_scores, '(b r) -> b r', r=repeats)
        # select repeat with highest score
        best_repeat = masks_scores.argmax(dim=1)
        best_masks = masks[torch.arange(b), best_repeat]
        audio = self.apply_mask(audio, best_masks.unsqueeze(-1))
        return audio, best_masks

    def forward(self, mask, **kwargs):
        return self.network(mask)




class VAEBase(base):
    def __init__(self) -> None:
        super().__init__()
        if type(self) == VAEBase: raise Exception('VAEBase class is not meant to be instantiated directly. Use a subclass like VariationalAutoEncoder')

    def reparameterize(self, mu, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if eps is None else eps
        return mu + eps * std, eps
    
    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(device)
        lengths = batch['lengths'].to(device)
        audio_out, mu, logvar = self(audio, lengths)
        
        if lengths.min() != lengths.max():
            mask = torch.arange(audio.size(-1)).to(lengths.device) < lengths[:, None]
            audio = torch.masked_fill(audio, ~mask.unsqueeze(1), 0)

        loss = nn.functional.mse_loss(audio_out, audio, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / (logvar.shape[0] * logvar.shape[1] * logvar.shape[2])
        loss += self.kld_weight * KLD
        return loss, {'loss':loss, 'KLD':KLD}, audio_out
    

class VariationalAutoEncoder(VAEBase):
    def __init__(self, input_dim=80, hidden_dim=256, latent_dim=16, layers=5, kld_weight=0.01):
        super().__init__()
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ) 
                for _ in range(layers)
            ],
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        self.mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        self.logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU()
                ) 
                for _ in range(layers)
            ],
        )
        self.output = nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1)

    def encode(self, x, lengths=None, eps=None, counts=None):
        in_length = x.size(-1)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        if counts is not None: # when we want to produce multiple samples from the same input
            mu = torch.repeat_interleave(mu, counts, dim=0)
            logvar = torch.repeat_interleave(logvar, counts, dim=0)

        z, eps = self.reparameterize(mu, logvar, eps)

        downsampled_lengths = None
        if lengths is not None:
            if counts is not None: lengths = torch.repeat_interleave(lengths, counts, dim=0)
            out_length = x.size(-1)
            downsampled_lengths = (lengths * out_length + in_length - 1) // in_length
            mask = torch.arange(out_length).unsqueeze(0).to(downsampled_lengths.device) < downsampled_lengths.unsqueeze(-1)
            
            z = torch.masked_fill(z, ~mask.unsqueeze(1), 0)
            mu = torch.masked_fill(mu, ~mask.unsqueeze(1), 0)
            logvar = torch.masked_fill(logvar, ~mask.unsqueeze(1), 0)

        return z, mu, logvar, eps, downsampled_lengths



    def forward(self, x, lengths, eps=None, counts=None):
        in_length = x.size(-1)
        
        z, mu, logvar, eps, _ = self.encode(x, lengths, eps, counts)

        x = self.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        x = self.output(x)
        return x, mu, logvar
    
class VariableLengthResidual(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, x):
        original_length = x.size(-1)
        residual = x
        x = self.module(x)
        new_length = x.size(-1)
        if new_length != original_length:
            residual = torch.nn.functional.interpolate(residual, size=new_length, mode='linear', align_corners=False)
        return x + residual

class ConvolutionBlock(nn.Module):
    def __init__(
            self,
            kernel_size:int,
            stride:int,
            transposed:bool=False,
            hidden_dim:int=256,
            ) -> None:
        super().__init__()

        self.transposed = transposed
        DepthwiseConv = nn.ConvTranspose1d if transposed else nn.Conv1d
        self.network = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LayerNorm(hidden_dim),
            Rearrange('b t c -> b c t'),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1, stride=1),
            nn.GLU(dim=1),
            DepthwiseConv(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.network(x)
    
class GRULayer(nn.Module):
    def __init__(self, dim=256, layers=2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True, bidirectional=False)

    def forward(self, x, lengths=None):
        # x: B, C, T
        x = rearrange(x, 'b c t -> b t c')
        x = self.ln(x)
        if lengths is not None: x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        if lengths is not None: x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = rearrange(x, 'b t c -> b c t')
        return x

class PlaceholderVQ(nn.Module):
    '''Use this as an identity op in place of VQ when you don't want to use VQ'''
    def forward(self, x, lens=None):
        return x, None, torch.tensor(0.0, device=x.device, dtype=x.dtype) # vq_x, indices, vq_commit_loss


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1, one=1.0, min_val=2):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = one
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)

    lengths = lengths.clamp(min=min_val)
    return lengths.to(dtype=torch.int)


from vector_quantize_pytorch import VectorQuantize
class VQVariationalAutoEncoder(VAEBase):
    def __init__(
            self, 
            input_dim=80, 
            hidden_dim=256, 
            latent_dim=16, 
            layers=5, 
            codebook_size=256, 
            commitment_weight=0.0,
            use_vq=True,
            norm_type='gn'
        ):
        super().__init__()
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.use_vq = use_vq
        self.norm_type = norm_type

        assert norm_type in ['gn', 'bn'], 'norm_type must be either "gn" or "bn"'
        
        self.VQ = VectorQuantize(
            dim = latent_dim, 
            codebook_size=codebook_size, 
            decay=0.99, 
            commitment_weight=commitment_weight,
            kmeans_init = True,   # set to True
            kmeans_iters = 10,
            threshold_ema_dead_code = 1.0,
            use_cosine_sim = True,
            rotation_trick = True,
        ) if use_vq else PlaceholderVQ()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1),
            *[  
                VariableLengthResidual(
                    nn.Sequential(
                        nn.GroupNorm(hidden_dim//32, hidden_dim) if norm_type == 'gn' else nn.BatchNorm1d(hidden_dim),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.SiLU(),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1, groups=hidden_dim),
                    )
                ) 
                for _ in range(layers)
            ],
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        self.mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        
        self.rnn_in = GRULayer(hidden_dim, layers=1)
        self.rnn_out = GRULayer(hidden_dim, layers=1)
        
        self.latent_to_hidden = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, stride=1)
        self.decoder = nn.Sequential(
            *[
                VariableLengthResidual(
                    nn.Sequential(
                        nn.GroupNorm(hidden_dim//32, hidden_dim) if norm_type == 'gn' else nn.BatchNorm1d(hidden_dim),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.SiLU(),
                        nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1, groups=hidden_dim),
                    )
                ) 
                for _ in range(layers)
            ],
        )
        self.output = nn.Sequential(
            nn.GroupNorm(hidden_dim//32, hidden_dim) if norm_type == 'gn' else nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding='same'),
            nn.GLU(dim=1),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        )

    def calc_downsampled_length(self, length):
        length = torch.as_tensor(length)
        length = calc_length(length, all_paddings=1, kernel_size=7, stride=2, ceil_mode=True, repeat_num=self.layers, one=1.0, min_val=2)
        return length

    def encode(self, x, lengths=None, counts=None):
        in_length = x.size(-1)
        x = self.encoder(x)
        
        if lengths is not None:
            downsampled_lengths = (lengths * x.size(-1) + in_length - 1) // in_length
        else: downsampled_lengths = None

        x = self.rnn_in(x, downsampled_lengths) + x

        z = self.mu(x)

        z, indices, closs = self.VQ(z.transpose(-1, -2), lens=downsampled_lengths)
    
        z = z.transpose(-1, -2)  


        if counts is not None: # when we want to produce multiple samples from the same input
            z = torch.repeat_interleave(z, counts, dim=0)

        if lengths is not None:
            if counts is not None: lengths = torch.repeat_interleave(lengths, counts, dim=0)
            out_length = x.size(-1)
            downsampled_lengths = (lengths * out_length + in_length - 1) // in_length
            mask = torch.arange(out_length).unsqueeze(0).to(downsampled_lengths.device) < downsampled_lengths.unsqueeze(-1)
            
            z = torch.masked_fill(z, ~mask.unsqueeze(1), 0)

        return z, indices, closs, downsampled_lengths


    def forward(self, x, lengths, counts=None):
        in_length = x.size(-1)
        
        z, _, closs, down_lengths = self.encode(x, lengths, counts)
        
        x = self.latent_to_hidden(z)
        x = self.rnn_out(x, down_lengths) + x
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        x = self.output(x)
        return x, z, closs
    
    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(device).to(torch.float32).squeeze(1)
        lengths = batch['lengths'].to(device)

        audio_pred, zvq, closs = self(audio, lengths)
        
        if kwargs.get('wandb', False) and random.random() < 0.05:
            with torch.no_grad():
                mask0 = audio[0]
                sample = audio_pred[0]
                vis = plt.imshow(sample[..., :lengths[0].item()].detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='magma', vmin=-1, vmax=1)
                wandb.log({'generated':vis})
                plt.close()
                vis = plt.imshow(mask0[..., :lengths[0].item()].detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='magma', vmin=-1, vmax=1)
                wandb.log({'target':vis})
                plt.close()

        loss = nn.functional.mse_loss(audio_pred, audio, reduction='none')
    
        pad_mask = torch.arange(audio.size(-1)).to(lengths.device) < lengths[:, None]
        if lengths.min() != lengths.max():
            loss = torch.masked_fill(loss, ~pad_mask.unsqueeze(1), 0)

        loss = loss.sum() / (pad_mask.sum() * audio.size(1))
        
        total_loss = loss + closs
        

        return total_loss, {'loss':loss, 'commit_loss':closs}, audio_pred

class BinaryVariationalAutoEncoder(VQVariationalAutoEncoder):
    def forward_pass(self, batch, device, **kwargs):
        masks = batch['masks'].to(device).to(torch.float32).squeeze(1)
        print(masks.shape)
        lengths = batch['lengths'].to(device)

        masks_out, zvq, closs = self(masks, lengths)
        masks_out_probs = torch.sigmoid(masks_out)
        
        if kwargs.get('wandb', False) and random.random() < 0.05:
            with torch.no_grad():
                mask0 = masks[0]
                sample = torch.bernoulli(masks_out_probs[0]) if random.random() < 0.5 else torch.round(masks_out_probs[0], decimals=0)
                vis = plt.imshow(sample[..., :lengths[0].item()].detach().cpu().numpy(), aspect='auto', cmap='gray')
                wandb.log({'generated':vis})
                plt.close()
                vis = plt.imshow(mask0[..., :lengths[0].item()].detach().cpu().numpy(), aspect='auto', cmap='gray')
                wandb.log({'target':vis})
                plt.close()

        prob_of_masks = masks_out_probs * masks + (1 - masks_out_probs) * (1 - masks)

        log_prob_of_masks = torch.log(prob_of_masks + 1e-8)

        pad_mask = torch.arange(masks.size(-1)).to(lengths.device) < lengths[:, None]
        if lengths.min() != lengths.max():
            log_prob_of_masks = torch.masked_fill(log_prob_of_masks, ~pad_mask.unsqueeze(1), 0)

        loss = -torch.sum(log_prob_of_masks) 
        loss = loss / (pad_mask.sum() * masks.size(1))
        
        total_loss = loss + closs
        

        return total_loss, {'loss':loss, 'commit_loss':closs}, masks_out_probs
    

class SingleStateVariationalAutoEncoder(VAEBase):
    def __init__(self, input_dim=80, hidden_dim=128, latent_dim=256, layers=6, kld_weight=0.000002, min_input_size=256):
        super().__init__()
        self.d_model = hidden_dim
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.min_input_size = min_input_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ) 
                for _ in range(layers)
            ],
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
        )
        self.mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)
        self.logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, stride=1)

        self.lstm_enc = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.lst_dec = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True, bidirectional=False)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, stride=1),
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU()
                ) 
                for _ in range(layers)
            ],
        )
        self.output = nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1)


    
    def bottleneck(self, x, lengths, mode='autoenc'):
        t = x.size(-1)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm_enc(x)
      
        h_n = rearrange(h_n[-1], 'b c -> b 1 c')

        if mode == 'compress':
            return h_n

        h_n = repeat(h_n, 'b 1 c -> b t c', t=t)
    
        x = torch.nn.utils.rnn.pack_padded_sequence(h_n, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lst_dec(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=t)
        x = rearrange(x, 'b t c -> b c t')
        return x


    def forward(self, x, lengths=None, mode='autoenc', eps=None):
        assert mode in ['autoenc', 'compress'], 'mode must be either autoenc or compress. compress returns the latent representation without decoding'
        in_length = x.size(-1)
        if lengths == None: lengths = torch.tensor([in_length] * x.size(0)).to(x.device)

        pad = 0
        if in_length < self.min_input_size:
            pad = self.min_input_size - in_length
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), pad).to(x.device)), dim=-1)

        x = self.encoder(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        z,_ = self.reparameterize(mu, logvar, eps)

        out_length = x.size(-1)
        downsampled_lengths = (lengths * out_length + in_length - 1) // in_length
        if lengths is not None:
            mask = torch.arange(out_length).unsqueeze(0).to(downsampled_lengths.device) < downsampled_lengths.unsqueeze(-1)
            z = torch.masked_fill(z, ~mask.unsqueeze(1), 0)
            mu = torch.masked_fill(mu, ~mask.unsqueeze(1), 0)
            logvar = torch.masked_fill(logvar, ~mask.unsqueeze(1), 0)
        
        z = self.bottleneck(z, downsampled_lengths, mode=mode)
        
        if mode == 'compress': return z

        x = self.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        x = self.output(x)
        if pad > 0: x = x[:,:,:-pad]
        return x, mu, logvar
    

class ConditionalFrequencyMaskingRanker(TrainableFrequencyMaskingRanker):
    def __init__(
            self,
            latent_dim=256,
            mel_bins=80,
            vae_config:Dict={},
            vae_state_dict_path:str=None,
            zero_masking=True, 
            loss_type='mse'
        ) -> None:
        super().__init__(zero_masking)
        assert loss_type in ['mse', 'mult']
        self.loss_type = loss_type
        self.mel_bins = mel_bins
        
        self.vae = SingleStateVariationalAutoEncoder(**vae_config)

        if vae_state_dict_path is not None: 
            vae_state_dict = torch.load(vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.vae.load_state_dict(vae_state_dict)
            print(f'Loaded VAE state dict from {vae_state_dict_path}')

        self.encode_mask = SwiGlu(self.mel_bins, output_dim=self.mel_bins, expansion_factor=1)
        self.encode_audio = nn.Sequential(
            SwiGlu(input_dim=latent_dim, output_dim=latent_dim, expansion_factor=1),
            nn.LayerNorm(latent_dim)
        )
        
        self.predict = nn.Sequential(
            SwiGlu(input_dim=latent_dim + self.mel_bins, output_dim=1, expansion_factor=2),
            Rearrange('b 1 -> b')
        )


    def learnt_augmentation(self, audio, repeats=1, lengths=None):
        b, c, t = audio.shape
        assert c == self.mel_bins, f'audio must have {self.mel_bins} channels'
        masks = torch.ones(b * repeats, c, 1).to(audio.device)

        masks = self.masker(masks).squeeze(-1)
        mask_scores = self(audio.repeat(repeats, 1, 1), masks, lengths=lengths)
        masks = rearrange(masks, '(b r) c -> b r c', r=repeats)
        masks_scores = rearrange(mask_scores, '(b r) -> b r', r=repeats)
        # select repeat with highest score
        best_repeat = masks_scores.argmax(dim=1)
        best_masks = masks[torch.arange(b), best_repeat]
        audio = self.apply_mask(audio, best_masks.unsqueeze(-1))
        
        return audio, best_masks

    def forward(self, audio, mask, lengths=None, counts=None):
        with torch.no_grad(): z_audio = rearrange(self.vae(audio, lengths=lengths, mode='compress'), 'b 1 c -> b c')

        z_audio = self.encode_audio(z_audio)
        z_mask = self.encode_mask(mask)
     

        if counts is not None: z_audio = torch.repeat_interleave(z_audio, counts, dim=0)
        
        z = torch.cat((z_audio, z_mask), dim=-1)
        return self.predict(z)


class GenerativePolicy(Policy):
    def __init__(            
            self,
            mel_bins=80,
            output_dim=80,
            hidden_dim=256,
            vae_config:Dict={},
            vae_state_dict_path:str=None,
            min_input_size=160,
            eps_clip=0.2,
            interpolate_mode='nearest-exact',
            freeze_encoder=True
        ) -> None:
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.vae = VariationalAutoEncoder(**vae_config)
        self.freeze_encoder = freeze_encoder

        if vae_state_dict_path is not None:
            vae_state_dict = torch.load(vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.vae.load_state_dict(vae_state_dict)
            print(f'Loaded VAE state dict from {vae_state_dict_path}')

        self.output = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1)

        
        self.hidden_dim = hidden_dim
        self.mel_bins = mel_bins
        self.output_dim = output_dim
        self.min_input_size = min_input_size
        self.eps_clip = eps_clip

    def forward(self, x, lengths, eps=None, counts=None):
        if x.size(-1) < self.min_input_size:
            pad = self.min_input_size - x.size(-1)
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), pad).to(x.device)), dim=-1)
         
        else: pad = 0
        
        in_length = x.size(-1)

        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, mu, logvar, eps = self.vae.encode(x, lengths, eps, counts)

        # x = self.output(z)        

        x = self.vae.decoder(z)
        x = torch.nn.functional.interpolate(x, size=in_length, mode='linear', align_corners=False)
        if pad > 0: x = x[:,:,:-pad]
        x = self.output(x)
   
        return x, eps


class ConditionalMaskingPolicy(GenerativePolicy):
    @torch.no_grad()
    def augment(self, audio, eps=None, lengths=None, *args, **kwargs):
        predictions, eps = self(audio, lengths, eps)
        probs = predictions.sigmoid() # 0 = mask, 1 = no mask
        mask = torch.bernoulli(probs)
    
        audio = audio * mask

        return audio, mask, {'probs':probs, 'eps':eps}
    
    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(dtype=torch.float32, device=device)
        lengths = batch['lengths'].to(device)
        ds_lengths = batch['ds_lengths'].to(device)
        #ds_lengths = batch['ds_lengths'].to(device)
        masks = batch['masks'].to(device, dtype=torch.long).squeeze(1)
        rewards = batch['rewards'].to(device)
        old_probs = batch['probs'].to(dtype=torch.float32, device=device).squeeze(1)
        eps = batch['eps'].to(dtype=torch.float32, device=device).squeeze(1)
        counts = batch['counts'].to(device)
        
        out, _, _, _ = self(audio, lengths, eps=eps, counts=counts)
        probs = out.sigmoid()
   
        entropy = torch.distributions.Bernoulli(probs).entropy()
        
        p_of_mask = probs * masks + (1 - probs) * (1 - masks)
        log_p_of_mask = torch.log(p_of_mask + 1e-8)


        old_p_of_mask = old_probs * masks + (1 - old_probs) * (1 - masks)
        old_log_p_of_mask = torch.log(old_p_of_mask + 1e-8)
        
        ratios = torch.exp(log_p_of_mask - old_log_p_of_mask)

        surr1 = ratios * rewards[:,None,None]
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * rewards[:,None,None]

        loss = -torch.min(surr1, surr2) 


        ds_lengths = ds_lengths.repeat_interleave(counts, dim=0)
        pad_mask = torch.arange(masks.size(-1)).to(ds_lengths.device) >= ds_lengths[:, None]
        loss = torch.masked_fill(loss, pad_mask.unsqueeze(1), 0)
        entropy = torch.masked_fill(entropy, pad_mask, 0)

        loss = loss.sum() / ((~pad_mask).sum() * self.output_dim)
        entropy = entropy.sum() / ((~pad_mask).sum() * self.output_dim)

        total_loss = loss - 0.0 * entropy
      
        return total_loss, {'loss':loss, 'entropy':entropy, 'total_loss':total_loss}
    


class timestep_encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Linear(1, hidden_dim // 2)

    def forward(self, t):
        proj = self.network(t)
        t = torch.cat((torch.sin(proj), torch.cos(proj)), dim=-1)
        return t


class MHA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            SwiGlu(input_dim=dim, output_dim=dim, expansion_factor=1)
        )

    def forward(self, x):
        residual = x
        x = self.ln(x.transpose(-1, -2))
        x = rearrange(x, 'b t c -> t b c')
        x = self.mha(x, x, x)[0]
        x = rearrange(x, 't b c -> b c t')
        x = self.proj(x)
        x = x + residual
        residual = x
        x = self.out_mlp(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + residual
        return x

class DTConditionalMaskingPolicy(Policy): 
    def __init__(            
            self,
            mel_bins=80,
            output_dim=80,
            latent_dim=32,
            hidden_dim=256,
            audio_vae_config:Dict={},
            audio_vae_state_dict_path:str=None,
            mask_vae_config:Dict={},
            mask_vae_state_dict_path:str=None,
            min_input_size=160,
            default_conditioning_reward=1.0,
            **kwargs
        ) -> None:
        super().__init__()
        self.default_conditioning_reward = default_conditioning_reward

        self.audio_enc = VariationalAutoEncoder(**audio_vae_config)
        if audio_vae_state_dict_path is not None:
            audio_vae_state_dict = torch.load(audio_vae_state_dict_path, map_location='cpu')['model_state_dict']
            #audio_vae_state_dict = {k.replace('vae.', 'audio_enc.'):v for k,v in audio_vae_state_dict.items()} 
            self.audio_enc.load_state_dict(audio_vae_state_dict)
            print(f'Loaded VAE state dict from {audio_vae_state_dict_path}')

        self.mask_enc = BinaryVariationalAutoEncoder(**mask_vae_config)
        if mask_vae_state_dict_path is not None:
            mask_vae_state_dict = torch.load(mask_vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.mask_enc.load_state_dict(mask_vae_state_dict)
            print(f'Loaded VAE state dict from {mask_vae_state_dict_path}')

        self.codebook_size = self.mask_enc.VQ.codebook_size
        self.embeddings = nn.Embedding(self.codebook_size + 1, hidden_dim, padding_idx=self.codebook_size)
        self.score_enc = timestep_encoder(hidden_dim=hidden_dim)

        self.encoder = nn.GRU(
            input_size=audio_vae_config['latent_dim'],
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
        )
        self.prediction = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.codebook_size + 1)
        )

        

    @torch.no_grad()
    def encode_audio(self, audio, lengths, eps=None, counts=None):
        z, _, _, _, downsampled_lengths = self.audio_enc.encode(audio, lengths, eps, counts)
        return z, downsampled_lengths
    
    @torch.no_grad()
    def encode_mask(self, mask, lengths):
        _, idx, _ = self.mask_enc.encode(mask, lengths)
        idx[idx == -1] = self.codebook_size
        return idx
    
    def encode(self, audio_emb, lengths):
        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(input=audio_emb.transpose(-1, -2), lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed_emb)
        return h_n
    
    def decode(self, h_n, mask_emb, lengths):
        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(input=mask_emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.decoder(packed_emb, h_n)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.prediction(x)
        return x
    
    
    @torch.no_grad()
    def generate(self, audio_hn, sample=True, target_output_length=None, target_prediction_steps=None):
        score_emb = self.score_enc(torch.tensor([self.default_conditioning_reward])[:, None].float().to(audio_hn.device)).unsqueeze(1)
        outputs = []
        h_n = audio_hn
        seq = score_emb

        total_steps = 80 if target_prediction_steps is None else target_prediction_steps

        for step in range(total_steps):
            z, h_n = self.decoder(seq, h_n)
            pred = self.prediction(z)
            
            if target_prediction_steps is not None:
                pred[..., self.codebook_size] = -torch.finfo(pred.dtype).max # prevent EOS token from being selected

            probs = pred.softmax(-1)
            if sample:
                idx = torch.multinomial(probs.squeeze(1), 1)
            else: idx = probs.argmax(-1)
            outputs.append(idx.squeeze().item())
            if idx.item() == self.codebook_size: break
            seq = self.embeddings(idx)
            
        outputs = [el for el in outputs if el != self.codebook_size]
        if len(outputs) == 0: return False
        print(outputs)
        outputs = torch.tensor(outputs, device=audio_hn.device)
       
        mask_latent = self.mask_enc.VQ.codebook[outputs][None].transpose(-1, -2)

        mask_h = self.mask_enc.decoder(mask_latent)

        if target_output_length is not None:
            mask_h = torch.nn.functional.interpolate(mask_h, size=target_output_length, mode='linear', align_corners=False)


        mask_pred = self.mask_enc.output(mask_h).sigmoid()
        mask_pred = torch.round(mask_pred, decimals=0).to(dtype=torch.long)

        return mask_pred

    @torch.no_grad()
    def augment(self, audio, *args, **kwargs):
        assert audio.size(0) == 1, 'only batch size 1 is supported atm'
        mode = self.training
        self.eval()

        lengths = torch.tensor([audio.size(-1)]).to(audio.device)
        audio_latent, downsampled_lengths = self.encode_audio(audio, lengths=lengths)
        h_n = self.encode(audio_latent, downsampled_lengths)
        mask_pred = self.generate(
            audio_hn=h_n,
            sample=True,
            target_output_length=lengths[0].item(),
            target_prediction_steps=downsampled_lengths[0].item()
        )
        audio = audio * mask_pred

        if mode: self.train()
        
        return audio, mask_pred



    def forward(self, audio_emb, downsampled_lengths, mask_emb, rewards, counts=None):
        h_n = self.encode(audio_emb, downsampled_lengths)
        if counts is not None: 
            h_n = torch.repeat_interleave(h_n, counts, dim=1)
            downsampled_lengths = downsampled_lengths.repeat_interleave(counts, dim=0)
        score_emb = self.score_enc(rewards[:, None].float()).unsqueeze(1)
        mask_emb = torch.cat((score_emb, mask_emb), dim=1)
        downsampled_lengths = downsampled_lengths + 1
        predictions = self.decode(h_n, mask_emb, downsampled_lengths)
        misc = {'h_n':h_n, 'mask_emb':mask_emb}
        return predictions, downsampled_lengths, misc


    def forward_pass(self, batch, device, **kwargs):
        audio = batch['audio'].to(dtype=torch.float32, device=device)
        lengths = batch['lengths'].to(device)
        masks = batch['masks'].to(device, dtype=torch.float32).squeeze(1)
        rewards = batch['rewards'].to(device)
        eps = batch['eps'].to(dtype=torch.float32, device=device).squeeze(1) if 'eps' in batch else None
        counts = batch['counts'].to(device)


        audio_latent, downsampled_lengths = self.encode_audio(audio, lengths=lengths, counts=None)
        mask_idx = self.encode_mask(masks, lengths.repeat_interleave(counts, dim=0))
        mask_emb = self.embeddings(mask_idx)

        predictions, downsampled_lengths, misc = self(audio_latent, downsampled_lengths, mask_emb, rewards, counts=counts)
        
        mask_idx = torch.cat((mask_idx, torch.full((mask_idx.size(0), 1), -100, dtype=torch.long, device=device)), dim=-1)
        # set last index to codebook size using downsampling lengths

        last_idx = torch.arange(mask_idx.size(0)).to(device)
        mask_idx[last_idx, downsampled_lengths - 1] = self.codebook_size
        lengths_mask = torch.arange(mask_idx.size(-1)).to(device) >= downsampled_lengths[:, None]

        ce_loss = nn.functional.cross_entropy(
            input = predictions.transpose(-1, -2),
            target= mask_idx,
            ignore_index=-100,
            reduction='none'
        )
        ce_loss = ce_loss.masked_fill(lengths_mask, 0)
        ce_loss = ce_loss.sum() / ((~lengths_mask).sum())

        if kwargs.get('wandb', False) and random.random() < 0.05:
            h_n = misc['h_n'][:, 0, None]
            mask_pred = self.generate(h_n, sample=True, target_prediction_steps=(downsampled_lengths[0] - 1).item(), target_output_length=lengths[0].item())
            if mask_pred is not False:
                vis = plt.imshow(mask_pred[0].detach().cpu().numpy(), aspect='auto', cmap='gray')
                wandb.log({'generated':vis})
                plt.close()


      
        return ce_loss, {'loss':ce_loss, 'total_loss':ce_loss}
    

class UnconditionalMaskGenerator(Policy): 
    def __init__(            
            self,
            hidden_dim=256,
            mask_vae_config:Dict={},
            mask_vae_state_dict_path:str=None,
            **kwargs
        ) -> None:
        super().__init__()

        self.mask_enc = BinaryVariationalAutoEncoder(**mask_vae_config)
        if mask_vae_state_dict_path is not None:
            mask_vae_state_dict = torch.load(mask_vae_state_dict_path, map_location='cpu')['model_state_dict']
            self.mask_enc.load_state_dict(mask_vae_state_dict)
            print(f'Loaded VAE state dict from {mask_vae_state_dict_path}')
            self.mask_enc.eval()

        self.codebook_size = self.mask_enc.VQ.codebook_size
        self.embeddings = nn.Embedding(self.codebook_size + 1, hidden_dim, padding_idx=self.codebook_size)

 
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
        )
        self.prediction = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.codebook_size + 1)
        )

    def total_parameters(self):
        params_decoder =  sum(p.numel() for p in self.decoder.parameters())
        params_prediction = sum(p.numel() for p in self.prediction.parameters())
        params_embeddings = sum(p.numel() for p in self.embeddings.parameters())
        total = params_decoder + params_prediction + params_embeddings
        return total
    
    @torch.no_grad()
    def encode_mask(self, mask, lengths):
        z, idx, closs, _ = self.mask_enc.encode(mask, lengths)
        downsampled_lengths = self.mask_enc.calc_downsampled_length(lengths)
        idx[idx == -1] = self.codebook_size
        return idx, downsampled_lengths, z
        
    def encode(self, audio_emb, lengths):
        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(input=audio_emb.transpose(-1, -2), lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed_emb)
        return h_n
    
    def decode(self, mask_emb, lengths):
        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(input=mask_emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.decoder(packed_emb)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.prediction(x)
        return x
    
    
    @torch.no_grad()
    def generate(self, sample=True, target_output_length=None, target_prediction_steps=None, device='cpu'):
        bos_emb = self.embeddings(torch.LongTensor([self.codebook_size]).to(device))
        bos_emb = rearrange(bos_emb, '1 c -> 1 1 c') # b, t, c
        outputs = []
        seq = bos_emb
        h_n = None
        total_steps = 80 if target_prediction_steps is None else target_prediction_steps

        mode = self.training
        self.eval()

        for step in range(total_steps):
            z, h_n = self.decoder(seq, h_n)
            pred = self.prediction(z)
            
            if target_prediction_steps is not None:
                pred[..., self.codebook_size] = -torch.finfo(pred.dtype).max # prevent EOS token from being selected

            probs = pred.softmax(-1)
            if sample:
                idx = torch.multinomial(probs.squeeze(1), 1)
            else: idx = probs.argmax(-1)
            outputs.append(idx.squeeze().item())
            if idx.item() == self.codebook_size: break
            seq = self.embeddings(idx)
            
        outputs = [el for el in outputs if el != self.codebook_size]
        if len(outputs) == 0: return False
        print(outputs)
        outputs = torch.tensor(outputs, device=device)
       
        mask_latent = self.mask_enc.VQ.codebook[outputs][None].transpose(-1, -2)
        mask_h = self.mask_enc.latent_to_hidden(mask_latent)
        mask_h = self.mask_enc.rnn_out(mask_h) + mask_h
        mask_h = self.mask_enc.decoder(mask_h)

        if target_output_length is not None:
            mask_h = torch.nn.functional.interpolate(mask_h, size=target_output_length, mode='linear', align_corners=False)

        mask_pred = self.mask_enc.output(mask_h).sigmoid()
        #mask_pred = torch.bernoulli(mask_pred).to(dtype=torch.long)
        mask_pred = torch.round(mask_pred, decimals=0).to(dtype=torch.long)

        if mode: 
            self.train()
            self.mask_enc.eval()

        return mask_pred

    @torch.no_grad()
    def augment(self, audio, *args, **kwargs):
        raise NotImplementedError
        assert audio.size(0) == 1, 'only batch size 1 is supported atm'
        mode = self.training
        self.eval()

        lengths = torch.tensor([audio.size(-1)]).to(audio.device)
        audio_latent, downsampled_lengths = self.encode_audio(audio, lengths=lengths)
        h_n = self.encode(audio_latent, downsampled_lengths)
        mask_pred = self.generate(
            audio_hn=h_n,
            sample=True,
            target_output_length=lengths[0].item(),
            target_prediction_steps=downsampled_lengths[0].item()
        )
        audio = audio * mask_pred

        if mode: self.train()
        
        return audio, mask_pred



    def forward(self, mask_emb, downsampled_lengths, counts=None):
        bos_emb = self.embeddings(torch.LongTensor([self.codebook_size]).to(mask_emb.device))
        
        bos_emb = repeat(bos_emb, '1 c -> b 1 c', b=mask_emb.size(0)) # b, t, c   
        mask_emb = torch.cat((bos_emb, mask_emb), dim=1)
        downsampled_lengths = downsampled_lengths + 1
        predictions = self.decode(mask_emb, downsampled_lengths)
        misc = {'mask_emb':mask_emb}
        return predictions, downsampled_lengths, misc


    def forward_pass(self, batch, device, **kwargs):
        #audio = batch['audio'].to(dtype=torch.float32, device=device)
        lengths = batch['lengths'].to(device)
        masks = batch['masks'].to(device, dtype=torch.float32).squeeze(1)
        #rewards = batch['rewards'].to(device)
        #eps = batch['eps'].to(dtype=torch.float32, device=device).squeeze(1) if 'eps' in batch else None
        #counts = batch['counts'].to(device)


        self.mask_enc.eval()
    
        mask_idx, downsampled_lengths, mask_z = self.encode_mask(masks, lengths)#.repeat_interleave(counts, dim=0))
        mask_emb = self.embeddings(mask_idx)

        predictions, downsampled_lengths, misc = self(mask_emb, downsampled_lengths, counts=None)
        
        mask_idx = torch.cat((mask_idx, torch.full((mask_idx.size(0), 1), -100, dtype=torch.long, device=device)), dim=-1)
        # set last index to codebook size using downsampling lengths

        last_idx = torch.arange(mask_idx.size(0)).to(device)
        mask_idx[last_idx, downsampled_lengths - 1] = self.codebook_size
        lengths_mask = torch.arange(mask_idx.size(-1)).to(device) >= downsampled_lengths[:, None]

        ce_loss = nn.functional.cross_entropy(
            input = predictions.transpose(-1, -2),
            target= mask_idx,
            ignore_index=-100,
            reduction='none'
        )
        ce_loss = ce_loss.masked_fill(lengths_mask, 0)
        ce_loss = ce_loss.sum() / ((~lengths_mask).sum())

        if kwargs.get('wandb', False) and random.random() < 0.025:
            mask_pred = self.generate(sample=True, target_prediction_steps=downsampled_lengths[0].item(), target_output_length=lengths[0].item(), device=device)
            if mask_pred is not False:
                vis = plt.imshow(mask_pred[0].detach().cpu().numpy(), aspect='auto', cmap='gray')
                wandb.log({'generated':vis})
                plt.close()
            vis = plt.imshow(masks[0, :, :lengths[0]].detach().cpu().numpy(), aspect='auto', cmap='gray')
            wandb.log({'original':vis})
            plt.close()

            mask_h = self.mask_enc.latent_to_hidden(mask_z[0, None])
            mask_h = self.mask_enc.rnn_out(mask_h) + mask_h
            mask_h = self.mask_enc.decoder(mask_h)

            mask_h = torch.nn.functional.interpolate(mask_h, size=lengths[0, None], mode='linear', align_corners=False)

            mask_pred = self.mask_enc.output(mask_h).sigmoid()
            mask_pred = torch.round(mask_pred, decimals=0).to(dtype=torch.long)
            #mask_pred = torch.bernoulli(mask_pred).to(dtype=torch.long)
            vis = plt.imshow(mask_pred[0].detach().cpu().numpy(), aspect='auto', cmap='gray')
            wandb.log({'reconstructed original':vis})
            plt.close()




        return ce_loss, {'loss':ce_loss, 'total_loss':ce_loss}



class AdditivePolicy(Policy):
    def __init__(
            self,
            input_dim=80,
            hidden_dim=80,
            output_dim=20,
            expansion_factor=2,
        ) -> None:
        super().__init__()

        hidden_dim = hidden_dim
        downsample_kernel = 7
        gated_kernel = (9,9)
        
        self.downsample_kernel = downsample_kernel
        self.gated_kernel = gated_kernel
        self.input_dim = input_dim
        self.block_layers = 5
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encode = (
                nn.Sequential(
                    *[
                        ResidualBlock(nn.Sequential(
                            GatedConv1d(
                                input_dim=hidden_dim, 
                                output_dim=hidden_dim, 
                                expansion_factor=expansion_factor, 
                                kernel_size=gated_kernel, 
                                stride=(1,1), 
                                padding=("same", "same"),
                            ),
                            BatchRenorm1d(hidden_dim) 
                        ))   for _ in range(self.block_layers)
                    ],
                    Rearrange('b c t -> b t c'),
                    nn.Linear(hidden_dim,hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim,output_dim*input_dim),
            ))
    

    def forward(self, x):
        x = self.encode(x)
        return x    
    
    def augment(self, audio, sample=True, return_probs=False, use_random=False, lengths=None):

        if not use_random:
            pred = self(audio) # b, c, t
            pred = rearrange(pred, 'b t (c p) -> b t c p', p=self.output_dim)
            pred = pred.softmax(-1)
        else:
            b, c, t = audio.shape
            pred = (torch.zeros(b, t, self.input_dim, self.output_dim) + 1/self.output_dim).to(audio.device)
        
        b = pred.shape[0]
        pred = rearrange(pred, 'b t c p-> (b t c) p')

        
        if sample: indexes = torch.multinomial(pred, 1).squeeze(-1)
        else: indexes = pred.argmax(-1)
        if return_probs: probs = torch.gather(pred, -1, indexes.unsqueeze(-1)).squeeze(-1)
        
        values = torch.round(-0.5 + indexes.float() * (2.0 / (self.output_dim - 1)), decimals=1)
        noise = rearrange(values, '(b t c) -> b c t', b=b, c=self.input_dim)
        
        if lengths is not None and lengths.min() != lengths.max():
            b, _, t = noise.shape
            pad_mask = torch.arange(t).to(lengths.device) >= lengths[:, None]
            noise = torch.masked_fill(noise, pad_mask.unsqueeze(1), 0)

        augmented_audio = audio + noise
        
        if return_probs: return augmented_audio, noise, probs
        else: return augmented_audio, noise
    
    def discretize(self, float_mask):
        indices = torch.round((float_mask + 0.5) * ((self.output_dim - 1) / 2.0)).long()
        indices = torch.clamp(indices, 0, self.output_dim - 1)

        return indices
    

policy_dict['FrequencyMaskingRanker'] = FrequencyMaskingRanker
policy_dict['MixedMaskingRanker'] = MixedMaskingRanker
policy_dict['UnconditionalFrequencyMaskingRanker'] = UnconditionalFrequencyMaskingRanker
policy_dict['ConditionalFrequencyMaskingRanker'] = ConditionalFrequencyMaskingRanker
policy_dict['UnconditionalMaskGenerator'] = UnconditionalMaskGenerator

policy_dict['VQVAE'] = VQVariationalAutoEncoder
policy_dict['BVAE'] = BinaryVariationalAutoEncoder
policy_dict['NoAugmentation'] = NoAugmentationPolicy
policy_dict['AdditivePolicy'] = AdditivePolicy
policy_dict['ConditionalMaskingPolicy'] = ConditionalMaskingPolicy
policy_dict['DTConditionalMaskingPolicy'] = DTConditionalMaskingPolicy
policy_dict['default'] = AdditivePolicy