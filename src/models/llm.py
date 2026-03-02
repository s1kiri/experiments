import torch
import torch.nn as nn
from .base_model import BaseModel


class HFLLM(torch.nn.Module, BaseModel):
    def __init__(self, model_name, trainable, max_length):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.trainable = trainable
        self.max_length = max_length

        for p in self.model.parameters():
            p.requires_grad = trainable

        cfg = self.model.config
        # Support both GPT-2 style (n_embd/n_layer) and LLaMA/general style (hidden_size/num_hidden_layers)
        self.hidden_size = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'n_embd', 768)
        self.num_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layer', 12)
        self.num_heads = getattr(cfg, 'num_attention_heads', None) or getattr(cfg, 'n_head', 12)

    def forward(self, inputs_embeds, attention_mask, labels=None):
        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
        )

    def generate(self, inputs_embeds, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

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
