from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange, repeat
from lcasr.components.batchrenorm import BatchRenorm1d
from torch.distributions import Normal
import random
from typing import Tuple, Callable, Dict
from einops.layers.torch import Rearrange


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
    
class SequenceToState(base):
    def __init__(
        self,
        input_dim:int,
        ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        torch.nn.init.normal_(self.query, mean=0, std=1e-1)
        self.keyvalue_proj = nn.Linear(input_dim, input_dim*2)
        self.out_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x:Tensor):
        query = self.query.expand(x.size(0), -1, -1)
        key, value = self.keyvalue_proj(x).chunk(2, dim=-1)
        attn = torch.nn.functional.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)
        return self.out_layer(torch.bmm(attn, value))


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


def cutout(spec, seq_len, num_rectangles=5, max_width=100, max_height=10):
    '''
    cutout_val: 'mean', 'mean_recording', 'zero'
    assumes a batch size of 1 (rearange to (F, B*N) if batch size > 1)
    '''
    if num_rectangles == 0: return spec

    spec_n = spec.shape[-1]
    ratio = spec_n / seq_len
    num_rectangles = int(num_rectangles * ratio) # if this spectrogram is shorter than the sequence lentgth used for tuning reduce the number of rectangles
    
    mask = torch.ones_like(spec)

    widths = torch.randint(1, max_width, (num_rectangles,))
    heights = torch.randint(1, max_height, (num_rectangles,))
    start_positions_x = torch.randint(0, spec.shape[-1], (num_rectangles,))
    end_positions_x = (start_positions_x + widths).clamp(max=spec.shape[-1])
    start_positions_y = torch.randint(0, spec.shape[-2], (num_rectangles,))
    end_positions_y = (start_positions_y + heights).clamp(max=spec.shape[-2])

    mask_values = []
    for i in range(num_rectangles):
        mask_values.append(spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]].mean())

    for i in range(num_rectangles):
        spec[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]] = mask_values[i]
        mask[:, start_positions_y[i]:end_positions_y[i], start_positions_x[i]:end_positions_x[i]].zero_()
    return spec, mask

from lcasr.utils.augmentation import SpecAugment 


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

    return audio_spec, mask_spec

class Policy(base):
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
    
    def discretize(self, float_mask, lengths=None):
        indices = torch.round((float_mask + 0.5) * ((self.output_dim - 1) / 2.0)).long()
        indices = torch.clamp(indices, 0, self.output_dim - 1)

        return indices
    
# val = 16
# values = round(-0.5 + val * (2.0 / 19.0),3)  # 
# values = torch.tensor(values)
# indices = torch.round((values + 0.5) * (19.0 / 2.0))
# indices = indices.long()  # Convert to integer indices
# indices = torch.clamp(indices, 0, 19)  # Ensure indices are within [0, 19]
# indices


# class Policy(base):
#     def __init__(
#             self,
#             input_dim=80,
#             hidden_dim=80,
#             output_dim=5,
#             expansion_factor=1,
#         ) -> None:
#         super().__init__()

#         hidden_dim = hidden_dim
#         downsample_kernel = 7
#         gated_kernel = (13,13)
        
#         self.downsample_kernel = downsample_kernel
#         self.gated_kernel = gated_kernel
#         self.input_dim = input_dim
#         self.mode = 'best' # 'best' | 'worst' | 'random' - how to select the mask acording to the scores
#         self.block_layers = 1    
#         self.hidden_dim = hidden_dim
#         mask_with_audio_dim = input_dim * 2# concatenated

#         self.masks_encode = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, stride=1, padding=0)
#         self.encode = nn.Sequential(
#             DepthWiseSeparableConv1d(input_dim=mask_with_audio_dim, output_dim=hidden_dim, kernel_size=downsample_kernel, stride=2),
#             nn.ReLU(),
#             BatchRenorm1d(hidden_dim),
#             DepthWiseSeparableConv1d(input_dim=hidden_dim, kernel_size=downsample_kernel, stride=2),
#             nn.ReLU(),
#             BatchRenorm1d(hidden_dim),
#             DepthWiseSeparableConv1d(input_dim=hidden_dim, kernel_size=downsample_kernel, stride=2),
#             ResidualBlock(
#                 nn.Sequential(
#                     *[
#                         nn.Sequential(
#                             GatedConv1d(
#                                 input_dim=hidden_dim, 
#                                 output_dim=hidden_dim, 
#                                 expansion_factor=expansion_factor, 
#                                 kernel_size=gated_kernel, 
#                                 stride=(1,1), 
#                                 padding=("same", "same"),
#                             ),
#                             BatchRenorm1d(hidden_dim) 
#                         )   for _ in range(self.block_layers)
#                     ]
#                 )
#             ),
#             DepthWiseSeparableConv1d(input_dim=hidden_dim, kernel_size=downsample_kernel, stride=2),
#             ResidualBlock(
#                 nn.Sequential(
#                     *[
#                         nn.Sequential(
#                             GatedConv1d(
#                                 input_dim=hidden_dim, 
#                                 output_dim=hidden_dim, 
#                                 expansion_factor=expansion_factor, 
#                                 kernel_size=gated_kernel, 
#                                 stride=(1,1), 
#                                 padding=("same", "same"),
#                             ),
#                             BatchRenorm1d(hidden_dim) 
#                         )   for _ in range(self.block_layers)
#                     ]
#                 )
#             ),
#             DepthWiseSeparableConv1d(input_dim=hidden_dim, kernel_size=downsample_kernel, stride=2),
#             ResidualBlock(
#                 nn.Sequential(
#                     *[
#                         nn.Sequential(
#                             GatedConv1d(
#                                 input_dim=hidden_dim, 
#                                 output_dim=hidden_dim, 
#                                 expansion_factor=expansion_factor, 
#                                 kernel_size=gated_kernel, 
#                                 stride=(1,1), 
#                                 padding=("same", "same"),
#                             ),
#                             BatchRenorm1d(hidden_dim) 
#                         )   for _ in range(self.block_layers)
#                     ]
#                 )
#             ),
#             Rearrange('b c t -> b t c'),
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim,output_dim),
#         )


#     @torch.no_grad()
#     def augment(self, data, masks=None, repeats=100):
#         if masks == None:
#             augmented_data, masks = augmentation_function(data, repeats)
#         else:
#             repeats = masks.shape[0]
#             augmented_data = data.repeat(repeats, 1, 1) + masks

#         if self.mode == 'random':
#             selected_mask = masks[0][None]
#             augmented_data = augmented_data[0][None]
#         else:
#             mask_scores = self(data.repeat(repeats, 1, 1), masks.unsqueeze(1)).mean((1)).softmax(-1)[:,:1].sum(-1)
            
#             if self.mode == 'best':
#                 selected_mask = masks[mask_scores.argmax()]
#                 augmented_data = augmented_data[mask_scores.argmax(), None]
#             elif self.mode == 'worst':
#                 selected_mask = masks[mask_scores.argmin()]
#                 augmented_data = augmented_data[mask_scores.argmin(), None]
#             else:
#                 raise ValueError(f"Unknown mode: {self.mode}")

#         return augmented_data, selected_mask.unsqueeze(0)
    

#     def forward(self, data, masks, counts=None):
#         x = data
#         if counts is not None: x = x.repeat_interleave(counts, dim=0)
        
#         masks = rearrange(masks, 'b 1 c t -> b (1 c) t').to(x.dtype)
#         masks = self.masks_encode(masks)
        
#         x = torch.cat((x, masks), dim=1)
   
#         x = self.encode(x)

#         return x



"""
    def __init__(
            self,
            input_dim=80,
            output_dim=80,
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.mode = 'best' # 'best' | 'worst' | 'random' - how to select the mask acording to the scores

        self.masks_encode = nn.Conv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        self.encode = nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim+output_dim, out_channels=input_dim*2, kernel_size=7, stride=2, groups=input_dim),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=1),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=7, stride=2, groups=input_dim*2),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=1),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=7, stride=2, groups=input_dim*2),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=1),
            ResidualBlock(
                nn.Sequential(
                    GatedConv1d(input_dim=input_dim*2, output_dim=input_dim*2, expansion_factor=1, kernel_size=(31,31), stride=(1,1), padding=("same", "same")),
                    BatchRenorm1d(input_dim*2),
                )
            ),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=7, stride=2, groups=input_dim*2),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=1),
            ResidualBlock(
                nn.Sequential(
                    GatedConv1d(input_dim=input_dim*2, output_dim=input_dim*2, expansion_factor=1, kernel_size=(31,31), stride=(1,1), padding=("same", "same")),
                    BatchRenorm1d(input_dim*2),
                )
            ),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=7, stride=2, groups=input_dim*2),
            torch.nn.Conv1d(in_channels=input_dim*2, out_channels=input_dim*2, kernel_size=1),
            ResidualBlock(
                nn.Sequential(
                    GatedConv1d(input_dim=input_dim*2, output_dim=input_dim*2, expansion_factor=1, kernel_size=(31,31), stride=(1,1), padding=("same", "same")),
                    BatchRenorm1d(input_dim*2),
                )
            ),
            Rearrange('b c t -> b t c'),
            nn.LayerNorm(input_dim*2),
            nn.Linear(input_dim*2,1),
        )
"""