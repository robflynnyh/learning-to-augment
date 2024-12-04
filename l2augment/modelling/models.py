from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange
#from lcasr.components.batchrenorm import BatchRenorm1d
from torch.distributions import Normal
import random

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



class Policy(nn.Module):
    def __init__(
            self,
            input_dim,
            masks_path,
            **kwargs
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

        masks = ~torch.load(masks_path, map_location='cpu')
        self.register_buffer('masks', masks)
        n_masks = masks.shape[0]

        self.network = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=4,
            batch_first=True,
            dropout=0.0
        )

        self.out_mask = nn.Sequential(
            nn.Linear(
                input_dim,
                n_masks * 2
            )
        )
        #init_small_output(self.out_mask[1])
        self.in_layer = nn.Sequential(
            SwiGlu(16*80, output_dim=input_dim-80, expansion_factor=1),
            nn.LayerNorm(input_dim-80)
        )

        #torch.bernoulli(torch.randn(16384, output_dim).sigmoid())

        self.register_buffer('lrs', torch.tensor([0.0, 0.001, 0.1, 0.5, 1.0, 2., 10, 100])) # 8

    @staticmethod
    def select_mask(mask_means, masks_stds, samples=100):
        dist = Normal(mask_means, torch.sqrt(masks_stds))
        samples = dist.sample(torch.tensor([samples]))
        max_idx = samples.argmax().item()
        
        return max_idx%samples.shape[-1]
        # if random.random() > 0.05:
        #     return mask_means.argmax().item()
        # else:
        #     return random.randint(0, mask_means.shape[0]-1)
        

    @torch.no_grad()
    def augment(self, data, state=None, prev_mask=None, **kwargs):
        seed = data
        if seed.size(-1) < 4096:
            pad = torch.zeros((seed.size(0), seed.size(1), 4096-seed.size(-1)))
            seed = torch.cat([seed, pad.to(seed.device)], dim=-1)
        elif seed.size(-1) > 4096:
            raise Exception("Exceeded max utterance length!")        
        seed = torch.nn.functional.avg_pool1d(seed, kernel_size=256, stride=256)
        seed = rearrange(seed, 'b c t -> b (c t)')
        assert seed.size(0) == 1, 'only implemented for b=1'
        mask_means, mask_stds, out_state, rnn_state = self.forward_mask(seed=seed.unsqueeze(1), state=state, prev_mask=prev_mask)
        mask_idx = self.select_mask(mask_means.squeeze(), mask_stds.squeeze())
        selected_mask = self.masks[mask_idx]
        augmented_data = data * selected_mask[None,:,None]

        output = {'augmented_data': augmented_data, 'state':rnn_state}
        output['seed'] = seed
        output['mask'] = torch.tensor([mask_idx])
        # output['selected_lr_probs'] = selected_probs
        # output['selected_lrs'] = selected_lrs
        # output['selected_lr_indexes'] = lr_selection_idxs
        return output

        
    def forward_mask(self, seed, state=None, prev_mask=None):
        b,l,c = seed.shape
        x = self.in_layer(seed)
        if prev_mask == None:
            prev_mask = torch.zeros((x.shape[0], x.shape[1], 80), device=x.device)
        x = torch.cat((prev_mask, x), dim=-1)
        x, hn = self.network(x, state)
        mask_means, mask_stds = torch.chunk(self.out_mask(x), 2, -1)
        mask_stds = mask_stds.abs()
        return mask_means, mask_stds, x, hn
    
    

class Value(nn.Module):
    def __init__(
            self,
            input_dim,
            masks_path,
            **kwargs
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

        masks = ~torch.load(masks_path, map_location='cpu')
        self.register_buffer('masks', masks)
        n_masks = masks.shape[0]

        self.network_a = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )
        self.network_b = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )
        self.combine = nn.Linear(input_dim+80, input_dim)

        self.out_mask = nn.Sequential(
            nn.Linear(
                input_dim,
                n_masks * 2
            )
        )
        #init_small_output(self.out_mask[1])
        self.in_layer = nn.Sequential(
            SwiGlu(16*80, output_dim=input_dim, expansion_factor=1),
            nn.LayerNorm(input_dim)
        )

        #torch.bernoulli(torch.randn(16384, output_dim).sigmoid())

        self.register_buffer('lrs', torch.tensor([0.0, 0.001, 0.1, 0.5, 1.0, 2., 10, 100])) # 8

    def forward(self, seed, state=None, prev_mask=None):
        b,l,c = seed.shape
        x = self.in_layer(seed)
        # if prev_mask == None:
        #     prev_mask = torch.zeros((x.shape[0], x.shape[1], 80), device=x.device)
        # x = torch.cat((prev_mask, x), dim=-1)
        x_a, hn_a  = self.network(x, state)
        mask_means, mask_stds = torch.chunk(self.out_mask(x), 2, -1)
        mask_stds = mask_stds.abs()
        return mask_means, mask_stds, x, hn
