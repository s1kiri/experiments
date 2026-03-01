import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class BaseMapper(torch.nn.Module, BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.trainable = True
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, z, attention_mask=None):
        return self.linear(z)

    def forward_flops(self):
        return 2 * self.in_dim * self.out_dim


class MLPMapper(torch.nn.Module, BaseModel):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim)
        self.trainable = True
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, z, attention_mask=None):
        z = self.act(self.linear1(z))
        z = self.act(self.linear2(z))
        z = self.linear3(z)
        return z

    def forward_flops(self):
        return 2 * self.in_dim * self.out_dim


class PooledMLPMapper(torch.nn.Module, BaseModel):
    """Masked chunk-mean-pools S tokens → n_ctx_tokens, then applies MLPMapper.

    attention_mask [B, S]: 1 for real tokens, 0 for padding.
    Padding positions contribute 0 to the pool mean, so short source texts
    don't pollute the soft-prompt representation with padding noise.

    n_ctx_tokens is the main experimental axis:
        1  → single global mean embedding (fastest, start here)
        4  → four regional embeddings
        8  → eight chunks
        16 → sixteen chunks (richest context, closer to original 512-token setup)
    """

    def __init__(self, in_dim, out_dim, n_ctx_tokens: int = 1):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.mlp = MLPMapper(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trainable = True

    def forward(self, z, attention_mask=None):
        """
        Args:
            z:               [B, S, D_in]   embedder output
            attention_mask:  [B, S]         1=real token, 0=padding (optional)
        Returns:
            [B, n_ctx_tokens, D_out]
        """
        B, S, D = z.shape
        k = self.n_ctx_tokens

        if attention_mask is None:
            attention_mask = torch.ones(B, S, device=z.device, dtype=z.dtype)
        mask = attention_mask.float().unsqueeze(-1)  # [B, S, 1]

        # Pad S to a multiple of k (extra positions keep mask=0 → don't contribute)
        pad = (-S) % k
        if pad:
            z    = F.pad(z,    (0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, pad))

        S2 = z.size(1)
        z    = z.reshape(B, k, S2 // k, D)        # [B, k, chunk, D]
        mask = mask.reshape(B, k, S2 // k, 1)     # [B, k, chunk, 1]

        # Masked mean: sum over real tokens / count of real tokens (min 1)
        z = (z * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [B, k, D]

        return self.mlp(z)   # [B, k, D_out]

    def forward_flops(self):
        return self.mlp.forward_flops()
