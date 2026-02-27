from .base_model import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel


class HFEmbedder(torch.nn.Module, BaseModel):
    def __init__(self, model_name, trainable, max_length):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.trainable = trainable
        self.max_length = max_length

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

    def forward_flops(self):
        # Standard transformer encoder FLOPs:
        # 2 * L * (4*d^2 + 2*d*T) * T
        d = self.hidden_size
        L = self.num_layers
        T = self.max_length

        flops = 2 * L * (4 * d * d + 2 * d * T) * T
        return flops
