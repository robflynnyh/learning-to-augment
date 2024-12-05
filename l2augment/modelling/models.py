from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange, repeat
from lcasr.components.batchrenorm import BatchRenorm1d
from torch.distributions import Normal
import random
from typing import Tuple, Callable, Dict

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


class Policy(base):
    def __init__(
            self,
            input_dim=80,
            hidden_dim=256,
            output_dim=80
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

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

        self.mask_prediction = nn.Sequential(
            nn.Linear(hidden_dim*2, output_dim*2)
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

    @torch.no_grad()
    def augment(self, data, state=None, cache=None):
        return_data = self.forward_sequential(data, state=state, cache=cache)
        masks = return_data['masks']
        masks = torch.repeat_interleave(masks, 8, dim=-1)
        data = data.repeat(2, 1, 1)
        masks = rearrange(masks, 'b (maskset c) t -> (b maskset) c t', maskset=2)   
        data = data * masks
        return_data['augmented_data'] = data
   
        return return_data
        
    def forward_sequential(self, data, state=None, cache=None):
        return_data = {}
        b,c,t = data.shape
        assert b == 1, 'only implemented for b=1'
        data = self.encode(data)
        if state is None:
            state = self.init_state.expand(-1, -1, data.size(2))
        else: 
            state = state.expand(-1, -1, data.size(2))

        data_with_state = torch.cat((state, data), dim=1).transpose(1, 2)
        mask_probs = self.mask_prediction(data_with_state).sigmoid().transpose(1, 2)
        return_data['mask_probabilities'] = mask_probs
        masks = torch.bernoulli(mask_probs)
        return_data['masks'] = masks
        
        data = torch.cat((data, masks), dim=1).transpose(1, 2)
        data = self.data_mask_combine(data)
        segment_embedding = self.sequence_to_state(data)
        new_state, new_cache = self.gru(segment_embedding, cache)

        return_data['next_state'] = new_state.transpose(1, 2)
        return_data['next_cache'] = new_cache

        return return_data
    
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
        states = torch.cat((initial_state, states), dim=-1)[:,:,:-1]
        states = rearrange(states, 'b d s -> (b s) d 1').expand(-1, -1, data.size(-1))
        data_with_state = torch.cat((states, data), dim=1).transpose(1, 2)
        mask_probs = self.mask_prediction(data_with_state).sigmoid().transpose(1, 2)
    
        assert mask_probs.shape == masks.shape, 'mask probs and masks should have the same shape'
        mask_probs = rearrange(mask_probs, '(b s) c t -> b s t c', s=s)
        return mask_probs


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

        # data_with_state_and_mask = torch.cat((data_with_state, masks.transpose(1, 2)), dim=-1)
        # data_with_state_and_mask = self.data_mask_state_combine(data_with_state_and_mask)

        # reward_given_state_and_mask = self.reward_prediction(data_with_state_and_mask)

        reward_given_state = rearrange(reward_given_state, '(b s) t d -> b s t d', s=s)
        reward_given_state_and_mask = rearrange(reward_given_state_and_mask, '(b s) t d -> b s t d', s=s)
        
        return reward_given_state, reward_given_state_and_mask
