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
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(
            input_dim,
            output_dim +8
        )
        #torch.bernoulli(torch.randn(16384, output_dim).sigmoid())

        self.register_buffer('lrs', torch.tensor([0.0, 0.001, 0.1, 0.5, 1.0, 2., 10, 100])) # 8

        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=80,
                out_channels=80,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=80,
                out_channels=256,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
        )


    # @staticmethod
    # def get_seed(batch_size, input_dim):
    #     return torch.normal(mean=0, std=1.0, size=(batch_size, input_dim))

    def augment(self, data, state=None, return_seed=False, return_mask=False):
        augmentation_probs, lr_probs, hn = self.forward(seed=data.unsqueeze(1), state=state)
        augmentation_mask = torch.distributions.Bernoulli(probs=augmentation_probs).sample().transpose(1,2)
        lr_selection_idxs = torch.multinomial(rearrange(lr_probs, 'b n c -> (b n) c'),num_samples=1)
        augmented_data = data * augmentation_mask
        output = {'augmented_data': augmented_data, 'state':hn}
        if return_seed: output['seed'] = data
        if return_mask: output['mask'] = augmentation_mask
     
        return output   
        
    def forward(self, seed, state=None):
        b,l,c,n = seed.shape
        seed = rearrange(seed, 'b l c n -> (b l) c n')
        seed = self.downsample(seed).mean(-1)
        seed = rearrange(seed, '(b l) c -> b l c', b=b, l=l)
        x, hn = self.network(seed, state)
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

