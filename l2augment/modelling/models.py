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

policy_dict = {}

class base(Module):
    def __init__(self) -> None:
        super().__init__()

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


def augmentation_function(audio, repeats=50):
    b, c, t = audio.shape

    audio_spec = audio.repeat(repeats, 1, 1)
    
    #random.seed(42)
    #torch.manual_seed(42)
    noise = torch.rand_like(audio_spec) * torch.rand_like(audio_spec)*2
    mask_spec = noise
    audio_spec = audio_spec + mask_spec

    # masks_spec = specaugmentation(torch.ones_like(audio_spec))
    # audio_spec = audio_spec * masks_spec + (1 - masks_spec) * audio_spec.mean().unsqueeze(0).unsqueeze(0)
        #print(masks.sum()/masks.numel(), 'soecmask')
   
    # audio_cutout, masks_cutout = cutout(
    #     rearrange(audio_cutout.clone(), 'b c t -> 1 c (b t)'),
    #     seq_len=t, 
    #     num_rectangles=160, 
    #     max_width=600, 
    #     max_height=50
    # )
    
    # audio_cutout = rearrange(audio_cutout, '1 c (b t) -> b c t', b=b*repeats)
    # masks_cutout = rearrange(masks_cutout, '1 c (b t) -> b c t', b=b*repeats)
    # audio = torch.cat((audio_spec, audio_cutout), dim=0)
    # masks = torch.cat((masks_spec, masks_cutout), dim=0)
    # return audio, masks


class Policy(base):
    def __init__(self) -> None:
        super().__init__()

    def augment(self, audio):
        raise NotImplementedError

class FrequencyMaskingRanker(Policy):
    def __init__(self, zero_masking=True) -> None:
        super().__init__()
        self.masker = SpecAugment(n_time_masks=0, n_freq_masks=6, freq_mask_param=34, zero_masking=True)
        self.zero_masking = zero_masking

    def apply_mask(self, audio, mask):
        if not self.zero_masking:
            audio = audio * mask + (1 - mask) * audio.mean(dim=(1,2), keepdim=True)
        else:
            audio = audio * mask
        return audio
        
    def augment(self, audio, use_random=False, repeats=1):
        assert audio.dim() == 3, 'audio must be 3D tensor'
        
        if use_random:
            mask_spec = self.masker(torch.ones_like(audio))
            
            audio = self.apply_mask(audio, mask_spec)
            
            assert mask_spec.shape[1] == 80, 'mask_spec must have 80 channels'

            return audio, mask_spec[:,:,0] # mask is same across all time steps so we can just return the first one
        else: 
            return self.learnt_augmentation(audio, repeats=repeats)
        
    def learnt_augmentation(self, audio, repeats=1):
        raise NotImplementedError # must be implemented in subclass
        
class UnconditionalFrequencyMaskingRanker(FrequencyMaskingRanker):
    def __init__(self, zero_masking=True, loss_type='mse') -> None:
        super().__init__(zero_masking)
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
        mask_scores = self.network(masks)
        masks = rearrange(masks, '(b r) c -> b r c', r=repeats)
        masks_scores = rearrange(mask_scores, '(b r) -> b r', r=repeats)
        # select repeat with highest score
        best_repeat = masks_scores.argmax(dim=1)
        best_masks = masks[torch.arange(b), best_repeat]
        audio = self.apply_mask(audio, best_masks.unsqueeze(-1))
        return audio, best_masks

    def forward(self, mask):
        return self.network(mask)
    
    def forward_pass(self, batch, device, **kwargs):
        masks = batch['masks'].to(device, dtype=torch.float32).squeeze(1) # B, 1, C -> B, C
        rewards = batch['rewards'].to(device)

        score = self(masks)

        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(score, rewards, reduction='mean')
        elif self.loss_type == 'mult':
            loss = -torch.mean(score * rewards)
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        
        return loss, {'loss':loss}

        
policy_dict['FrequencyMaskingRanker'] = FrequencyMaskingRanker
policy_dict['UnconditionalFrequencyMaskingRanker'] = UnconditionalFrequencyMaskingRanker


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
    
policy_dict['AdditivePolicy'] = AdditivePolicy
policy_dict['default'] = AdditivePolicy