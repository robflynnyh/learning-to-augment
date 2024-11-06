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
        a, b = self.in_layer(x).chunk(2)
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
            SwiGlu(input),
            SwiGlu(input, output_dim),
        )

    def forward(self, batch_size):
        x = torch.normal(mean=0, std=1.0, size=(batch_size, self.input_dim))
        x = self.network(x)
        x = x.sigmoid()
        return x
