import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Optional


# ---------------------------------------------------------------------------
# Shared pooling helper
# ---------------------------------------------------------------------------

def _chunk_mean_pool(
    z: torch.Tensor,
    k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked chunk-mean-pool [B, S, D] → [B, k, D].

    Splits S tokens into k equal-size chunks and averages each chunk,
    ignoring padding positions (mask=0).
    """
    B, S, D = z.shape
    if attention_mask is None:
        attention_mask = torch.ones(B, S, device=z.device, dtype=z.dtype)
    mask = attention_mask.float().unsqueeze(-1)  # [B, S, 1]

    pad = (-S) % k
    if pad:
        z    = F.pad(z,    (0, 0, 0, pad))
        mask = F.pad(mask, (0, 0, 0, pad))

    S2   = z.size(1)
    z    = z.reshape(B, k, S2 // k, D)
    mask = mask.reshape(B, k, S2 // k, 1)
    return (z * mask).sum(2) / mask.sum(2).clamp(min=1)  # [B, k, D]


# ---------------------------------------------------------------------------
# Private MLP kernel (used internally by MLPMapper and PooledMLPMapper)
# ---------------------------------------------------------------------------

class _MLPKernel(nn.Module):
    """Token-wise 3-layer MLP: [B, *, D_in] → [B, *, D_out].

    Attribute names (linear1/2/3) are kept identical to the old MLPMapper so
    that PooledMLPMapper checkpoints (mapper.mlp.linear1.*) remain loadable.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.act     = nn.SiLU()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self.act(self.linear2(self.act(self.linear1(x)))))


# ---------------------------------------------------------------------------
# Public mapper classes
# ---------------------------------------------------------------------------

class BaseMapper(torch.nn.Module, BaseModel):
    """Pool to k tokens via chunk-mean, then project with one linear layer.

    Lightest mapper: single linear on top of pooled embeddings.
    n_ctx_tokens=1  → single global mean + projection (fastest baseline).
    """

    def __init__(self, in_dim: int, out_dim: int, n_ctx_tokens: int = 1):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.linear       = nn.Linear(in_dim, out_dim)
        self.trainable    = True
        self.in_dim       = in_dim
        self.out_dim      = out_dim

    def forward(self, z: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.linear(_chunk_mean_pool(z, self.n_ctx_tokens, attention_mask))

    def forward_flops(self) -> int:
        return self.n_ctx_tokens * 2 * self.in_dim * self.out_dim


class MLPMapper(torch.nn.Module, BaseModel):
    """Pool to k tokens then apply 3-layer MLP.

    Fixed version of the old token-wise MLPMapper: the pooling step now
    compresses S encoder tokens → k context tokens before the MLP.
    """

    def __init__(self, in_dim: int, out_dim: int, n_ctx_tokens: int = 1):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.act          = nn.SiLU()
        self.linear1      = nn.Linear(in_dim, in_dim)
        self.linear2      = nn.Linear(in_dim, out_dim)
        self.linear3      = nn.Linear(out_dim, out_dim)
        self.trainable    = True
        self.in_dim       = in_dim
        self.out_dim      = out_dim

    def forward(self, z: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = _chunk_mean_pool(z, self.n_ctx_tokens, attention_mask)
        z = self.act(self.linear1(z))
        z = self.act(self.linear2(z))
        return self.linear3(z)

    def forward_flops(self) -> int:
        k, d, o = self.n_ctx_tokens, self.in_dim, self.out_dim
        return k * 2 * (d * d + d * o + o * o)


class PooledMLPMapper(torch.nn.Module, BaseModel):
    """Legacy mapper kept for checkpoint compatibility.

    Functionally identical to MLPMapper.  Existing checkpoints produced with
    PooledMLPMapper are still loadable because the internal weight tensors
    are stored under the same keys (mapper.mlp.linear1.*, etc.).

    Prefer MLPMapper for new experiments.
    """

    def __init__(self, in_dim: int, out_dim: int, n_ctx_tokens: int = 1):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.mlp          = _MLPKernel(in_dim, out_dim)
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.trainable    = True

    def forward(self, z: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mlp(_chunk_mean_pool(z, self.n_ctx_tokens, attention_mask))

    def forward_flops(self) -> int:
        k, d, o = self.n_ctx_tokens, self.in_dim, self.out_dim
        return k * 2 * (d * d + d * o + o * o)


class PerceiverMapper(torch.nn.Module, BaseModel):
    """Perceiver-style resampler mapper.

    k learnable query tokens cross-attend over encoder output (+ self-attend
    among themselves) via a stack of TransformerDecoderLayers.

    Unlike chunk-mean pooling, the queries are not constrained to contiguous
    segments and can attend to any position — making this more expressive.

    Architecture per layer:
        queries  →  self-attention among k queries
                 →  cross-attention: queries attend to encoder output
                 →  feed-forward (d → 4d → d)
    Final LayerNorm applied after all layers.

    Args:
        in_dim:        encoder hidden size
        out_dim:       LLM hidden size (must equal LLM d_model)
        n_ctx_tokens:  number of output context tokens (k)
        n_heads:       attention heads (out_dim must be divisible by n_heads)
        n_layers:      number of decoder layers (default 2)
    """

    def __init__(
        self,
        in_dim:        int,
        out_dim:       int,
        n_ctx_tokens:  int = 16,
        n_heads:       int = 8,
        n_layers:      int = 2,
    ):
        super().__init__()
        self.n_ctx_tokens = n_ctx_tokens
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.trainable    = True

        # Learnable query vectors initialised with small random values
        self.queries = nn.Parameter(torch.randn(1, n_ctx_tokens, out_dim) * 0.02)

        # Linear projection from encoder space → decoder space (identity if equal)
        self.in_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=out_dim,
                nhead=n_heads,
                dim_feedforward=out_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,   # pre-norm: more stable training
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        z:              torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:              [B, S, in_dim]  encoder output
            attention_mask: [B, S]          1=real token, 0=padding
        Returns:
            [B, n_ctx_tokens, out_dim]
        """
        B = z.size(0)
        memory = self.in_proj(z)                              # [B, S, out_dim]
        q      = self.queries.expand(B, -1, -1)               # [B, k, out_dim]

        # TransformerDecoderLayer expects key_padding_mask where True = IGNORE
        key_pad = (attention_mask == 0) if attention_mask is not None else None

        for layer in self.layers:
            q = layer(tgt=q, memory=memory, memory_key_padding_mask=key_pad)

        return self.norm(q)   # [B, k, out_dim]

    def forward_flops(self) -> int:
        k, d, L = self.n_ctx_tokens, self.out_dim, self.n_layers
        S = 512   # assumed encoder max_length (approximation)
        # Per-layer breakdown:
        #   cross-attn projections (Q from k queries, K/V from S memory tokens)
        cross_attn = 4 * (k + S) * d * d + 4 * k * S * d
        #   self-attn projections (Q/K/V/O for k queries)
        self_attn  = 8 * k * d * d
        #   FFN: k tokens through d→4d→d
        ffn        = 4 * k * d * (4 * d)
        return L * (cross_attn + self_attn + ffn)


# ---------------------------------------------------------------------------
# Legacy: AlignVLM mapper (large memory footprint, experimental)
# ---------------------------------------------------------------------------

class AlignVLMMapper(torch.nn.Module, BaseModel):
    """LEGACY — Align module from AlignVLM (arXiv:2502.01341 §3.1).

    Maps source embeddings to a convex combination of LLM token embeddings,
    ensuring mapper outputs lie in the convex hull of the LLM embedding space
    (in-distribution by construction).

    NOT recommended for new experiments:
    - Intermediate P_vocab tensor [B, k, V≈152K] ≈ 77 MB in bf16 per forward
    - Requires setup_llm() to be called before use

    Pipeline:
        z [B, S, d_emb]
          → chunk-mean-pool  →  [B, k, d_emb]
          → W1 (Linear)      →  [B, k, d_llm]
          → LN1              →  [B, k, d_llm]
          → lm_head (ref)    →  [B, k, V]   (W2 = shared LLM lm_head, no extra params)
          → LN2              →  [B, k, V]
          → softmax          →  P_vocab [B, k, V]
          → @ E_text (ref)   →  F_align [B, k, d_llm]

    W2 (lm_head) and E_text (embedding table) are shared references to LLM weights —
    gradients flow through them naturally when the LLM is trainable.

    Call setup_llm(llm_model) before use; MapperLLMModule does this automatically.
    """

    def __init__(self, in_dim: int, out_dim: int, n_ctx_tokens: int = 16):
        super().__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.n_ctx_tokens = n_ctx_tokens
        self.trainable    = True

        self.w1  = nn.Linear(in_dim, out_dim, bias=False)
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2           = None
        self._lm_head      = None
        self._embed_module = None

    def setup_llm(self, llm_model: nn.Module) -> None:
        """Bind LLM embedding table and lm_head references."""
        self._embed_module = llm_model.get_input_embeddings()
        self._lm_head      = llm_model.lm_head
        vocab_size         = self._embed_module.weight.size(0)
        self.ln2           = nn.LayerNorm(vocab_size)

    def forward(self, z: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self._lm_head is not None, (
            "AlignVLMMapper.setup_llm() was not called before forward(). "
            "Ensure MapperLLMModule calls it in __init__."
        )
        z = _chunk_mean_pool(z, self.n_ctx_tokens, attention_mask)  # [B, k, d_emb]

        x       = self.ln1(self.w1(z))                              # [B, k, d_llm]
        logits  = self._lm_head(x).float()                          # [B, k, V]
        logits  = self.ln2(logits)
        p_vocab = torch.softmax(logits, dim=-1)

        E = self._embed_module.weight                                # [V, d_llm]
        return torch.matmul(p_vocab.to(E.dtype), E)                 # [B, k, d_llm]

    def forward_flops(self) -> int:
        V   = self._embed_module.weight.size(0) if self._embed_module is not None else 152000
        k, d, D = self.n_ctx_tokens, self.in_dim, self.out_dim
        return (
            k * 2 * d * D    # W1
            + k * 2 * D * V  # W2 (lm_head)
            + k * 2 * V * D  # P_vocab @ E_text
        )
