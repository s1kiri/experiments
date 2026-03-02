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
        # Qwen3-style transformer: GQA attention + SwiGLU FFN
        cfg = self.model.config
        d         = self.hidden_size
        L         = self.num_layers
        T         = self.max_length
        kv_factor = getattr(cfg, 'num_key_value_heads', self.num_heads) / self.num_heads
        d_ff      = getattr(cfg, 'intermediate_size', 4 * d)

        # Attention: Q + K(GQA) + V(GQA) + O projections + QKᵀ + AV
        attn = 2 * T * d * d * (2 + 2 * kv_factor) + 4 * T * T * d
        # SwiGLU FFN: gate + up + down (3 matrices of shape d × d_ff)
        ffn  = 6 * T * d * d_ff
        return L * (attn + ffn)
    

# class QwenHFEmbedder(torch.nn.Module, BaseModel):
#     def __init__(self, model_name, trainable, max_length):
#         super().__init__()
#         from transformers import AutoModel
#         self.model = AutoModel.from_pretrained(model_name)
#         self.trainable = trainable
#         self.max_length = max_length

#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         self.hidden_size = self.model.config.hidden_size
#         self.num_layers = self.model.config.num_hidden_layers

#     def forward(self, input_ids, attention_mask):
#         output = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         last_hidden_states = output.last_hidden_state
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

#     def forward_flops(self):
#         # Standard transformer encoder FLOPs:
#         # 2 * L * (4*d^2 + 2*d*T) * T
#         d = self.hidden_size
#         L = self.num_layers
#         T = self.max_length

#         flops = 2 * L * (4 * d * d + 2 * d * T) * T
#         return flops
