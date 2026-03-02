import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Optional


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
        d, out = self.in_dim, self.out_dim
        # 3 linear layers: (d→d), (d→out), (out→out) on a single token vector
        return 2 * (d * d + d * out + out * out)


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
        k, d, out = self.n_ctx_tokens, self.in_dim, self.out_dim
        # 3 linear layers: (d→d), (d→out), (out→out) applied to k pooled tokens
        return k * 2 * (d * d + d * out + out * out)


class AlignVLMMapper(torch.nn.Module, BaseModel):
    """Align module from AlignVLM (arXiv:2502.01341 §3.1), adapted for text-to-text.

    Maps source embeddings to a convex combination of LLM token embeddings, ensuring
    mapper outputs lie in the convex hull of the LLM embedding space (in-distribution
    by construction).

    Pipeline:
        z [B, S, d_emb]
          → chunk-mean-pool  →  [B, k, d_emb]   (same as PooledMLPMapper)
          → W1 (Linear)      →  [B, k, d_llm]
          → LN1              →  [B, k, d_llm]
          → lm_head (ref)    →  [B, k, V]        (W2 = shared LLM lm_head, no extra params)
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
        # ln2 and LLM refs are bound in setup_llm() after resize_token_embeddings
        self.ln2           = None
        self._lm_head      = None
        self._embed_module = None

    def setup_llm(self, llm_model: nn.Module) -> None:
        """Bind LLM embedding table and lm_head references.

        Must be called after llm_model.resize_token_embeddings() so that ln2
        is created with the correct (post-resize) vocab_size.
        """
        self._embed_module = llm_model.get_input_embeddings()   # nn.Embedding [V, D]
        self._lm_head      = llm_model.lm_head                  # nn.Linear(D → V)
        vocab_size         = self._embed_module.weight.size(0)
        self.ln2           = nn.LayerNorm(vocab_size)

    def forward(self, z: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self._lm_head is not None, (
            "AlignVLMMapper.setup_llm() was not called before forward(). "
            "Ensure MapperLLMModule calls it in __init__."
        )
        B, S, _ = z.shape
        k = self.n_ctx_tokens

        # ── 1. Chunk-mean-pool (identical to PooledMLPMapper) ──────────────
        if attention_mask is None:
            attention_mask = torch.ones(B, S, device=z.device, dtype=z.dtype)
        mask = attention_mask.float().unsqueeze(-1)      # [B, S, 1]
        pad  = (-S) % k
        if pad:
            z    = F.pad(z,    (0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, pad))
        S2   = z.size(1)
        z    = z.reshape(B, k, S2 // k, z.size(-1))     # [B, k, chunk, d_emb]
        mask = mask.reshape(B, k, S2 // k, 1)
        z = (z * mask).sum(2) / mask.sum(2).clamp(min=1)  # [B, k, d_emb]

        # ── 2. W1 → LN1 ─────────────────────────────────────────────────────
        x = self.ln1(self.w1(z))                           # [B, k, d_llm]

        # ── 3. lm_head (W2) → LN2 → softmax ────────────────────────────────
        # Cast to fp32 before LN + softmax for numerical stability (V ≈ 150K).
        logits  = self._lm_head(x).float()                 # [B, k, V]
        logits  = self.ln2(logits)                         # [B, k, V]
        p_vocab = torch.softmax(logits, dim=-1)            # [B, k, V]

        # ── 4. Weighted sum of LLM token embeddings ──────────────────────────
        E = self._embed_module.weight                       # [V, d_llm]
        return torch.matmul(p_vocab.to(E.dtype), E)        # [B, k, d_llm]

    def forward_flops(self):
        V   = self._embed_module.weight.size(0) if self._embed_module is not None else 152000
        k, d, D = self.n_ctx_tokens, self.in_dim, self.out_dim
        return (
            k * 2 * d * D    # W1
            + k * 2 * D * V  # W2 (lm_head)
            + k * 2 * V * D  # P_vocab @ E_text
        )
