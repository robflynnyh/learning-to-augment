from torch.nn import Module
from torch import nn, Tensor
import torch
from einops import rearrange
from lcasr.components.batchrenorm import BatchRenorm1d

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
        self.out_layer = nn.Linear(input_dim*2, output_dim)
        self.act = nn.SiLU()
        
    def forward(self, x:Tensor):
        a, b = self.in_layer(x).chunk(2, dim=-1)
        c = a * self.act(b)
        return self.out_layer(c)


class Policy(base):
    def __init__(
            self,
            input_dim,
            output_dim,
            **kwargs
        ) -> None:
        super().__init__()
        self.input_dim = input_dim

        self.network = nn.RNN(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=5,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                output_dim +8
            )
        )
        self.in_layer = nn.Sequential(
            SwiGlu(16*80, output_dim=input_dim),
            nn.LayerNorm(input_dim)
        )
        #torch.bernoulli(torch.randn(16384, output_dim).sigmoid())

        self.register_buffer('lrs', torch.tensor([0.0, 0.001, 0.1, 0.5, 1.0, 2., 10, 100])) # 8


    # @staticmethod
    # def get_seed(batch_size, input_dim):
    #     return torch.normal(mean=0, std=1.0, size=(batch_size, input_dim))

    def augment(self, data, state=None, **kwargs):
        seed = data
        if seed.size(-1) < 4096:
            pad = torch.zeros((seed.size(0), seed.size(1), 4096-seed.size(-1)))
            seed = torch.cat([seed, pad.to(seed.device)])
        elif seed.size(-1) > 4096:
            raise Exception("Exceeded max utterance length!")        
        seed = torch.nn.functional.avg_pool1d(seed, kernel_size=256, stride=256)
        seed = rearrange(seed, 'b c t -> b (c t)')

        augmentation_probs, lr_probs, hn = self.forward(seed=seed.unsqueeze(1), state=state)
        augmentation_mask = torch.distributions.Bernoulli(probs=augmentation_probs).sample().transpose(1,2)
        lr_selection_idxs = torch.multinomial(rearrange(lr_probs, 'b n c -> (b n) c'),num_samples=1).squeeze(-1)
        selected_probs=torch.gather(lr_probs, dim=-1, index=lr_selection_idxs.unsqueeze(-1).unsqueeze(-1)).squeeze()
        selected_lrs = self.lrs[lr_selection_idxs]
        augmented_data = data * augmentation_mask
        output = {'augmented_data': augmented_data, 'state':hn}
        output['seed'] = seed
        output['mask'] = augmentation_mask
        output['selected_lr_probs'] = selected_probs
        output['selected_lrs'] = selected_lrs
        output['selected_lr_indexes'] = lr_selection_idxs
        return output   
        
    def forward(self, seed, state=None):
        b,l,c = seed.shape
        x = self.in_layer(seed)
        x, hn = self.network(x, state)
        x = self.out(x)
        mask_probs = x[:,:,:-8].sigmoid()
        lr_probs = x[:,:,-8:].softmax(-1)
        return mask_probs, lr_probs, hn
    

class Value(base):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_size,
            **kwargs
        ) -> None:
        super().__init__()
        
        self.bos_token = nn.Embedding(1, input_dim)
        self.backbone_network = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(
            hidden_size,
            output_dim
        )


    def forward(self, x:Tensor, state: Tensor = None): # x = action from policy
        batch_size = x.size(0)
        bos_token = self.bos_token.weight[None].repeat(batch_size, 1, 1)
        x, hn = self.backbone_network(
            torch.cat([bos_token, x], dim=1), 
            state
        )
        x = self.out(x)
        return x, hn

