"""Pipeline loading and inference utilities for mapper-LLM experiments.

Shared between infer.py (CLI) and infer.ipynb (notebook).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import yaml
from transformers import AutoTokenizer

from src.models.embedder import HFEmbedder
from src.models.mapper import *  # noqa: F401,F403  (exposes all mapper classes for globals())
from src.models.llm import HFLLM
from src.models.lightning_module import MapperLLMModule
from src.data.schema import SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_model(name: str, cache_dir: Optional[str]) -> str:
    """Return local cache path if available, otherwise the HF hub name."""
    if cache_dir:
        local = os.path.join(cache_dir, name.replace("/", "--"))
        if os.path.isdir(local):
            return local
    return name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_models(config_path: str):
    """Construct all pipeline components from a training config.

    Returns (cfg, emb_tok, llm_tok, embedder, mapper, llm).
    All model parameters are set to requires_grad=False (inference mode).
    Does NOT apply training-only setup (mean-emb init, gradient checkpointing).
    """
    cfg       = _load_config(config_path)
    cache_dir = cfg["models"].get("cache_dir") or None

    # ── Tokenizers ──────────────────────────────────────────────────────────
    emb_tok = AutoTokenizer.from_pretrained(
        _resolve_model(cfg["models"]["embedder"]["name"], cache_dir)
    )
    if emb_tok.pad_token_id is None:
        emb_tok.pad_token_id = emb_tok.eos_token_id

    llm_tok = AutoTokenizer.from_pretrained(
        _resolve_model(cfg["models"]["llm"]["name"], cache_dir)
    )
    llm_tok.pad_token = llm_tok.eos_token
    llm_tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # ── Models ───────────────────────────────────────────────────────────────
    embedder_cfg = cfg["models"]["embedder"]
    embedder = HFEmbedder(
        model_name=_resolve_model(embedder_cfg["name"], cache_dir),
        trainable=False,
        max_length=embedder_cfg["max_length"],
    )

    mapper_cfg   = cfg["models"]["mapper"]
    mapper_class = globals()[mapper_cfg["class"]]
    mapper       = mapper_class(**mapper_cfg["params"])
    mapper.trainable = False

    llm_cfg = cfg["models"]["llm"]
    llm = HFLLM(
        model_name=_resolve_model(llm_cfg["name"], cache_dir),
        trainable=False,
        max_length=llm_cfg["max_length"],
    )
    llm.model.resize_token_embeddings(len(llm_tok))

    return cfg, emb_tok, llm_tok, embedder, mapper, llm


def load_pipeline(
    checkpoint_path: str,
    config_path: str,
    device: str = "cpu",
) -> tuple[MapperLLMModule, object, object]:
    """Load a trained pipeline from a Lightning checkpoint.

    Args:
        checkpoint_path: path to a .ckpt file produced by ModelCheckpoint
        config_path:     path to the YAML config used during training
        device:          "cpu", "cuda:0", etc.

    Returns:
        (module, emb_tok, llm_tok)
        module is a MapperLLMModule in eval() mode, moved to `device`.
    """
    cfg, emb_tok, llm_tok, embedder, mapper, llm = build_models(config_path)

    module = MapperLLMModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        # These are not stored in hparams (ignored in save_hyperparameters),
        # so we must pass them explicitly.
        embedder=embedder,
        mapper=mapper,
        llm=llm,
        llm_tokenizer=llm_tok,
    )
    module = module.to(device).eval()
    return module, emb_tok, llm_tok


def predict(
    module: MapperLLMModule,
    emb_tok,
    llm_tok,
    records: list[dict],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    device: Optional[str] = None,
    repetition_penalty: float = 1.3,
) -> list[str]:
    """Run inference on a list of input records.

    Args:
        records:            list of dicts, each with keys:
                              source_text (str, required)
                              task        (str, "narrative" | "qa", required)
                              question    (str, optional, only for qa)
        max_new_tokens:     maximum tokens to generate per example
        batch_size:         number of examples to process in one GPU batch
        device:             device override; defaults to the module's current device
        repetition_penalty: passed to model.generate()

    Returns:
        list[str] of prediction strings, same order as records
    """
    if device is None:
        device = str(next(module.parameters()).device)
    module = module.to(device).eval()

    all_preds: list[str] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        all_preds.extend(
            _predict_batch(module, emb_tok, llm_tok, chunk,
                           max_new_tokens, device, repetition_penalty)
        )
    return all_preds


# ---------------------------------------------------------------------------
# Internal: single-batch inference
# ---------------------------------------------------------------------------

def _predict_batch(
    module: MapperLLMModule,
    emb_tok,
    llm_tok,
    records: list[dict],
    max_new_tokens: int,
    device: str,
    repetition_penalty: float,
) -> list[str]:
    source_texts = [r["source_text"] for r in records]
    tasks        = [r.get("task", "narrative") for r in records]
    questions    = [r.get("question", "") or "" for r in records]

    # ── Encode source texts ─────────────────────────────────────────────────
    enc = emb_tok(
        source_texts,
        padding=True,
        truncation=True,
        max_length=module.embedder.max_length,
        return_tensors="pt",
    )
    source_ids  = enc.input_ids.to(device)
    source_mask = enc.attention_mask.to(device)

    preds: list[str] = [""] * len(records)

    with torch.inference_mode():
        z = module.embedder(source_ids, source_mask)
        h = module._wrap_context(module.mapper(z, source_mask))  # [B, k+2, D]

        get_embeds = module.llm.model.get_input_embeddings()
        _bypass    = "<think>\n</think>\n\n"

        # Group by task so each group shares the same prefix template
        task_groups: dict[str, list[int]] = {}
        for i, t in enumerate(tasks):
            task_groups.setdefault(t, []).append(i)

        for task_name, indices in task_groups.items():
            h_group = h[indices]  # [G, k+2, D]

            if task_name == "qa":
                prefix_texts = [
                    f"{_bypass}<QA> {questions[i]} <ANSWER>" for i in indices
                ]
            else:
                prefix_texts = [f"{_bypass}<REPRODUCE>"] * len(indices)

            enc_p        = llm_tok(prefix_texts, add_special_tokens=False,
                                   padding=True, return_tensors="pt")
            prefix_ids   = enc_p.input_ids.to(device)
            prefix_mask  = enc_p.attention_mask.to(device)
            prefix_embs  = get_embeds(prefix_ids)

            gen_input = torch.cat([h_group, prefix_embs], dim=1)
            h_mask    = torch.ones(
                len(indices), h_group.size(1), device=device, dtype=torch.long
            )
            gen_mask  = torch.cat([h_mask, prefix_mask], dim=1)

            out = module.llm.generate(
                inputs_embeds=gen_input,
                attention_mask=gen_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=llm_tok.pad_token_id,
                eos_token_id=llm_tok.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

            for j, idx in enumerate(indices):
                preds[idx] = module._strip_thinking(
                    llm_tok.decode(out[j], skip_special_tokens=True)
                )

    return preds
