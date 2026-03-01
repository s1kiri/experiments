import torch
import torch.nn as nn
from .base_model import BaseModel


class BaseMapper(torch.nn.Module, BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.trainable = True
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, z):
        return self.linear(z)

    def forward_flops(self):
        return 2 * self.in_dim * self.out_dim
    
class MLPMapper(torch.nn.Module, BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, in_dim)
        self.linear2 = torch.nn.Linear(in_dim, out_dim)
        self.linear3 = torch.nn.Linear(out_dim, out_dim)
        self.trainable = True
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, z):
        activation = torch.nn.SiLU()
        z = activation(self.linear1(z))
        z = activation(self.linear2(z))
        z = self.linear3(z)
        return z

    def forward_flops(self):
        return 2 * self.in_dim * self.out_dim
