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
    

class Value(base):
    def __init__(
            self,
            input_dim,
            output_dim,
            **kwargs
        ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_size = kwargs.get("state_dim", 80)
        
        self.initial_state = nn.Parameter(torch.empty((1, self.state_size)))
        nn.init.uniform_(self.initial_state, a=-0.1, b=0.1)

        self.network = nn.Sequential(
            SwiGlu(input_dim + self.state_size, output_dim=input_dim),
            SwiGlu(input_dim, output_dim),
        )

    def get_initial_state(self, batch_size):
        return self.initial_state.repeat(batch_size, 1)

    def forward(self, x:Tensor, state: Tensor | None = None): # x = action from policy
        if state is None:
            state = self.get_initial_state(x.size(0))
        x = torch.cat([state, x], dim=-1)
        x = self.network(x)
        predicted_reward, next_state = x.split(split_size_or_sections=[1, self.state_size], dim=-1)
        return predicted_reward, next_state

