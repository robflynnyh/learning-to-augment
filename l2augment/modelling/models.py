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
        padding:Tuple[int]=(0,0)
        ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim == None else output_dim

        self.in_layer = nn.Conv1d(in_channels=input_dim, out_channels=input_dim*expansion_factor*2, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.out_layer = nn.Conv1d(in_channels=input_dim*expansion_factor, out_channels=output_dim, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])

    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=1)
        c = a * torch.nn.functional.silu(b)
        return self.out_layer(c)
    
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


class ResidualBlock(nn.Module):
    def __init__(
            self,
            module:nn.Module,
            ) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)



class Policy(base):
    def __init__(
            self,
            input_dim=80,
            output_dim=80,
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

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


        # self.encode = nn.Sequential(
        #     *[ResidualBlock(
        #         nn.Sequential(
        #             GatedConv1d(input_dim=input_dim, output_dim=input_dim, expansion_factor=2, kernel_size=(31,31), stride=(1,1), padding=("same", "same")),
        #             BatchRenorm1d(input_dim),
        #             # Rearrange('b c t -> b t c'),
        #             # nn.LayerNorm(80),
        #             # Rearrange('b t c -> b c t'),
        #         )
        #     ) for _ in range(5)], 
        #     nn.Conv1d(in_channels=input_dim, out_channels=output_dim*2, kernel_size=1, stride=1, padding=0),
        # )

        # self.predict = nn.Sequential(
        #     *[ResidualBlock(
        #         nn.Sequential(
        #             GatedConv1d(input_dim=output_dim, output_dim=output_dim, expansion_factor=1, kernel_size=(31,31), stride=(1,1), padding=("same", "same")),
        #             BatchRenorm1d(output_dim),
        #         )
        #     )
        #      for _ in range(2)],   
        #     nn.Conv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0),
        # )

    @torch.no_grad()
    def augment(self, data, augmentation_function, repeats=10):
        data = data.repeat(repeats, 1, 1)
        unmasked = torch.ones_like(data)
        mask = augmentation_function(unmasked).unsqueeze(1)
        mask_scores = self(data, mask).mean((1,2)).sigmoid()
        selected_mask = mask[mask_scores.argmin()]
        augmented_data = data[0, None] * selected_mask + (1 - selected_mask) * data[0, None].mean().unsqueeze(0).unsqueeze(0)

        return augmented_data, selected_mask

        # random_masks = torch.randn((100, 2,data.shape[1], data.shape[2]), device=data.device).sigmoid().bernoulli().to(torch.bool)
        
        # mask_probs = self(data, random_masks, counts=10)
        # mask_preds = mask_probs.mean((1,2))
        # mask = mask_preds.argmax()
        # mask = random_masks[mask]
    
        # data = repeat(data, 'b c t -> (maskset b) c t', maskset=2)
        # augmented_data = data * mask 
        # return {
        #     'augmented_data': augmented_data,
        #     'masks': mask,
        #     'mask_probs': mask_probs
        # }
       
        
    def forward(self, data, masks, counts=None):
        x = data
        if counts is not None:
            x = x.repeat_interleave(counts, dim=0)
        
        masks = rearrange(masks, 'b maskset c t -> b (maskset c) t', maskset=1).to(x.dtype)
        masks = self.masks_encode(masks)
        
        x = torch.cat((x, masks), dim=1)
   
        x = self.encode(x) # b (2 c) t
        # print(x.shape,'--')
        # x = rearrange(x, 'b (bool maskset c) t -> bool b maskset c t', maskset=2, bool=2)
        # if counts is not None:
        #     x = x.repeat_interleave(counts, dim=1)
        # x = (masks*x[0] + (~masks)*x[1])
        # x = rearrange(x, 'b maskset c t -> b (maskset c) t', maskset=2)
        # x = self.predict(x)
        return x

# class Policy(base):
#     def __init__(
#             self,
#             input_dim=80,
#             hidden_dim=256,
#             output_dim=80
#         ) -> None:
#         super().__init__()
#         self.input_dim = input_dim

#         self.encode = nn.Sequential(
#             GatedConv1d(
#                 input_dim=input_dim,
#                 output_dim=input_dim*2,
#                 kernel_size=(1,7),
#                 stride=(1,2),
#                 padding=(0,3)
#             ),
#             BatchRenorm1d(input_dim*2),
#             GatedConv1d(
#                 input_dim=input_dim*2,
#                 output_dim=hidden_dim,
#                 kernel_size=(1,7),
#                 stride=(1,2),
#                 padding=(0,3)
#             ), 
#             BatchRenorm1d(hidden_dim),
#             GatedConv1d(
#                 input_dim=hidden_dim,
#                 output_dim=hidden_dim,
#                 kernel_size=(1,7),
#                 stride=(1,2),
#                 padding=(0,3)
#             ), 
#             BatchRenorm1d(hidden_dim),
#         )
#         self.init_state = nn.Parameter(torch.randn(1, hidden_dim, 1))
#         nn.init.normal_(self.init_state, mean=0, std=1e-2)

#         self.mask_prediction = nn.Sequential(
#             nn.LayerNorm(hidden_dim*2),
#             nn.Linear(hidden_dim*2, output_dim*2)
#         )

#         self.data_mask_combine = nn.Sequential(
#             SwiGlu(hidden_dim+output_dim*2, hidden_dim, expansion_factor=1),
#             nn.LayerNorm(hidden_dim)
#         )
#         self.sequence_to_state = SequenceToState(hidden_dim)

#         self.gru = nn.GRU(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.0
#         )

#     @torch.no_grad()
#     def augment(self, data, state=None, cache=None):
#         return_data = self.forward_sequential(data, state=state, cache=cache)
#         masks = return_data['masks']
#         masks = torch.repeat_interleave(masks, 8, dim=-1)
#         data = data.repeat(2, 1, 1)
#         masks = rearrange(masks, 'b (maskset c) t -> (b maskset) c t', maskset=2)   
#         data = data * masks
#         return_data['augmented_data'] = data
   
#         return return_data
        
#     def forward_sequential(self, data, state=None, cache=None):
#         return_data = {}
#         b,c,t = data.shape
#         assert b == 1, 'only implemented for b=1'
#         data = self.encode(data)
#         if state is None:
#             state = self.init_state.expand(-1, -1, data.size(2))
#         else: 
#             state = state.expand(-1, -1, data.size(2))

#         data_with_state = torch.cat((state, data), dim=1).transpose(1, 2)
#         mask_probs = self.mask_prediction(data_with_state).sigmoid().transpose(1, 2)
#         return_data['mask_probabilities'] = mask_probs
#         masks = torch.bernoulli(mask_probs)
#         return_data['masks'] = masks
        
#         data = torch.cat((data, masks), dim=1).transpose(1, 2)
#         data = self.data_mask_combine(data)
#         segment_embedding = self.sequence_to_state(data)
#         new_state, new_cache = self.gru(segment_embedding, cache)

#         return_data['next_state'] = new_state.transpose(1, 2)
#         return_data['next_cache'] = new_cache

#         return return_data
    
#     def forward_parallel(self, data, masks):
#         """requires that masks are already generated and saved from the forward_sequential method during rollout"""
#         b, s, c, t = data.shape
#         data = rearrange(data, 'b s c t -> (b s) c t')
#         data = self.encode(data)
#         masks = rearrange(masks, 'b s c t -> (b s) c t')
#         data_with_mask = torch.cat((data, masks), dim=1).transpose(1, 2)
#         data_with_mask = self.data_mask_combine(data_with_mask)
#         segment_embedding = self.sequence_to_state(data_with_mask)
#         segment_embedding = rearrange(segment_embedding, '(b s) 1 d -> b s d', s=s)
        
#         states, _ = self.gru(segment_embedding)
#         states = states.transpose(1, 2)
#         initial_state = self.init_state.expand(states.size(0), -1, -1)
#         states = torch.cat((initial_state, states), dim=-1)[:,:,:-1]
#         states = rearrange(states, 'b d s -> (b s) d 1').expand(-1, -1, data.size(-1))
#         data_with_state = torch.cat((states, data), dim=1).transpose(1, 2)
        
#         mask_probs = self.mask_prediction(data_with_state)
       
#         mask_probs = mask_probs.sigmoid().transpose(1, 2)
#         assert mask_probs.shape == masks.shape, 'mask probs and masks should have the same shape'
#         mask_probs = rearrange(mask_probs, '(b s) c t -> b s t c', s=s)
#         return mask_probs


class Value(base):
    def __init__(
            self,
            input_dim=80,
            hidden_dim=256,
            output_dim=80,
            reward_dim=4096,
        ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.reward_dim = reward_dim

        self.encode = nn.Sequential(
            GatedConv1d(
                input_dim=input_dim,
                output_dim=input_dim*2,
                kernel_size=(1,7),
                stride=(1,2),
                padding=(0,3)
            ),
            BatchRenorm1d(input_dim*2),
            GatedConv1d(
                input_dim=input_dim*2,
                output_dim=hidden_dim,
                kernel_size=(1,7),
                stride=(1,2),
                padding=(0,3)
            ), 
            BatchRenorm1d(hidden_dim),
            GatedConv1d(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                kernel_size=(1,7),
                stride=(1,2),
                padding=(0,3)
            ), 
            BatchRenorm1d(hidden_dim),
        )
        self.init_state = nn.Parameter(torch.randn(1, hidden_dim, 1))
        nn.init.normal_(self.init_state, mean=0, std=1e-2)

        self.data_with_state_ds = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.reward_prediction = nn.Sequential(
            nn.Linear(hidden_dim, reward_dim)
        )

        self.data_mask_combine = nn.Sequential(
            SwiGlu(hidden_dim+output_dim*2, hidden_dim, expansion_factor=1),
            nn.LayerNorm(hidden_dim)
        )

     
        self.sequence_to_state = SequenceToState(hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )

        
   
    def forward_parallel(self, data, masks):
        """requires that masks are already generated and saved from the forward_sequential method during rollout"""
        b, s, c, t = data.shape
        data = rearrange(data, 'b s c t -> (b s) c t')
        data = self.encode(data)
        masks = rearrange(masks, 'b s c t -> (b s) c t')
        data_with_mask = torch.cat((data, masks), dim=1).transpose(1, 2)
        data_with_mask = self.data_mask_combine(data_with_mask)
        segment_embedding = self.sequence_to_state(data_with_mask)
        segment_embedding = rearrange(segment_embedding, '(b s) 1 d -> b s d', s=s)
        
        states, _ = self.gru(segment_embedding)
        states = states.transpose(1, 2)
        initial_state = self.init_state.expand(states.size(0), -1, -1)
        shifted_states = torch.cat((initial_state, states), dim=-1)[:,:,:-1]
        shifted_states = rearrange(shifted_states, 'b d s -> (b s) d 1').expand(-1, -1, data.size(-1))
        data_with_state = torch.cat((shifted_states, data), dim=1).transpose(1, 2)
        data_with_state = self.data_with_state_ds(data_with_state)

        reward_given_state = self.reward_prediction(data_with_state)

        states = rearrange(states, 'b d s -> (b s) 1 d', s=s).expand(-1, data_with_state.size(1), -1)
        data_with_cur_state = torch.cat((states, data_with_state), dim=-1)
        data_with_cur_state = self.data_with_state_ds(data_with_cur_state)
        reward_given_state_and_mask = self.reward_prediction(data_with_cur_state)

        reward_given_state = rearrange(reward_given_state, '(b s) t d -> b s t d', s=s)
        reward_given_state_and_mask = rearrange(reward_given_state_and_mask, '(b s) t d -> b s t d', s=s)
        
        return reward_given_state, reward_given_state_and_mask
