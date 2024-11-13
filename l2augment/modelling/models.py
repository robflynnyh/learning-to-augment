from torch.nn import Module
from torch import nn, Tensor
import torch

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

        self.network = nn.Sequential(
            SwiGlu(input_dim),
            SwiGlu(input_dim, output_dim),
        )

    @staticmethod
    def get_seed(batch_size, input_dim):
        return torch.normal(mean=0, std=1.0, size=(batch_size, input_dim))

    def augment(self, data, return_seed=False, return_mask=False):
        batch_size = data.size(0)
        seed = self.get_seed(batch_size=batch_size, input_dim=self.input_dim)
        augmentation_probs = self.forward(seed=seed)
        augmentation_mask = torch.bernoulli(augmentation_probs)[:,:,None]
  
        augmented_data = data * augmentation_mask
        output = {'augmented_data': augmented_data}
        if return_seed: output['seed'] = seed
        if return_mask: output['mask'] = augmentation_mask
     
        return output
        
    def forward(self, seed):
        x = self.network(seed)
        x = x.sigmoid()
        return x
    

class Value(base):
    def __init__(
            self,
            input_dim,
            output_dim,
            **kwargs
        ) -> None:
        super().__init__()

        self.backbone_network = nn.RNN(
            input_size=input_dim,
            hidden_size=input_dim*2,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(
            input_dim*2,
            output_dim
        )


    def forward(self, x:Tensor, state: Tensor | None = None): # x = action from policy
        x, hn = self.backbone_network(x, state)
        x = self.out(x)
        return x

