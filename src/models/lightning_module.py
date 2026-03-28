import torch
from collections import defaultdict
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup

from src.models.embedder import HFEmbedder
from src.models.llm import HFLLM
from src.models.mapper import BaseMapper
from src.metrics import bleu4, rouge_l

# Number of validation samples per task to display in TensorBoard
_VAL_LOG_EXAMPLES = 8


class MapperLLMModule(pl.LightningModule):
    def __init__(
        self,
        embedder: HFEmbedder,
        mapper: BaseMapper,
        llm: HFLLM,
        llm_tokenizer,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        target_metric: None = None,
        val_generate: bool = True,
        debug: bool = False,
        soft_prompt_tokens: int = 0,
    ):
        super().__init__()

        self.embedder = embedder
        self.mapper = mapper
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer

        # Give mappers that need LLM weight references (e.g. AlignVLMMapper) a
        # chance to bind them. setup_llm() is called here so vocab_size is
        # already correct (train.py calls resize_token_embeddings before __init__).
        if hasattr(mapper, 'setup_llm'):
            mapper.setup_llm(llm.model)

        # Soft-prompt: p learnable vectors prepended after _wrap_context(mapper output).
        # LLM base is frozen; full backward still flows through LLM to reach these params.
        self.soft_prompt_tokens = soft_prompt_tokens
        if soft_prompt_tokens > 0:
            self.soft_prompt = torch.nn.Parameter(
                torch.zeros(1, soft_prompt_tokens, llm.hidden_size)
            )
        else:
            self.soft_prompt = None

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.val_generate = val_generate
        self.debug = debug

        # stores (pred, answer, source_text, question) quads per task across validation steps
        self.val_storage = defaultdict(list)

        self.target_metric = target_metric
        self.reached_target = False
        self.compute_to_quality = None

        self.flops_per_step = 0  # set in on_fit_start

        # Precompute <CONTEXT> / </CONTEXT> token IDs for wrapping mapper output
        ctx_start_id = llm_tokenizer.convert_tokens_to_ids("<CONTEXT>")
        ctx_end_id   = llm_tokenizer.convert_tokens_to_ids("</CONTEXT>")
        self.register_buffer("ctx_start_id", torch.tensor([ctx_start_id]))
        self.register_buffer("ctx_end_id",   torch.tensor([ctx_end_id]))

        self.save_hyperparameters(ignore=["embedder", "mapper", "llm", "llm_tokenizer"])

    # -----------------------------
    # helpers
    # -----------------------------
    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 <think>…</think> blocks and stray tags.

        <think>/<</think> are vocabulary tokens, not special tokens, so
        skip_special_tokens=True in decode() does NOT strip them.
        The bypass prefix reduces thinking-block generation but doesn't
        eliminate it (QA context can re-trigger it after <ANSWER>).
        """
        import re
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = text.replace('<think>', '').replace('</think>', '')
        return text.strip()

    @staticmethod
    def _trunc_words(text: str, n: int) -> str:
        """Truncate to at most n whitespace-split tokens."""
        toks = text.split()
        return " ".join(toks[:n]) if len(toks) > n else text

    # -----------------------------
    # debug helpers
    # -----------------------------
    def _dbg(self, msg: str):
        """Print when debug=True."""
        if self.debug:
            print(f"[DEBUG] {msg}", flush=True)

    @staticmethod
    def _tensor_stats(t: torch.Tensor, name: str) -> str:
        """One-line summary of a tensor for debug output."""
        f = t.detach().float()
        return (
            f"{name}: shape={list(t.shape)} dtype={t.dtype} "
            f"mean={f.mean():.4f} std={f.std():.4f} "
            f"min={f.min():.4f} max={f.max():.4f}"
        )

    # -----------------------------
    # helpers
    # -----------------------------
    def _wrap_context(self, h: torch.Tensor) -> torch.Tensor:
        """Prepend <CONTEXT> and append </CONTEXT> embeddings around mapper output h.

        Args:
            h: [B, S, D_llm]
        Returns:
            [B, S+2, D_llm]
        """
        get_embeds = self.llm.model.get_input_embeddings()
        B = h.size(0)
        ctx_s = get_embeds(self.ctx_start_id.unsqueeze(0)).expand(B, -1, -1)  # [B, 1, D]
        ctx_e = get_embeds(self.ctx_end_id.unsqueeze(0)).expand(B, -1, -1)    # [B, 1, D]
        # Cast h to the embedding dtype (bf16 on GPU, fp32 on CPU) before concat
        h = h.to(ctx_s.dtype)
        return torch.cat([ctx_s, h, ctx_e], dim=1)  # [B, S+2, D]

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, batch):
        source_input_ids      = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]

        if self.debug:
            real  = source_attention_mask.sum().item()
            total = source_attention_mask.numel()
            self._dbg(
                f"source tokens: {real}/{total} real "
                f"({100*real/total:.0f}%) across {source_input_ids.size(0)} samples"
            )

        z = self.embedder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
        )
        self._dbg(self._tensor_stats(z, "embedder z"))

        h_raw = self.mapper(z, source_attention_mask)
        self._dbg(self._tensor_stats(h_raw, "mapper h_raw"))
        h = self._wrap_context(h_raw)
        # h: [B, n_ctx+2, D_llm]

        if self.soft_prompt is not None:
            sp = self.soft_prompt.expand(h.size(0), -1, -1).to(h.dtype)
            h = torch.cat([h, sp], dim=1)  # [B, n_ctx+2+p, D_llm]

        target_input_ids      = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]

        if self.debug:
            labels_raw = batch["labels"]
            active = (labels_raw != -100).sum().item()
            tot    = labels_raw.numel()
            self._dbg(f"labels: {active}/{tot} active ({100*active/tot:.1f}%)")
            tgt_real = target_input_ids[0][target_attention_mask[0].bool()]
            self._dbg(
                f"target[0] decoded: "
                f"{repr(self.llm_tokenizer.decode(tgt_real.tolist())[:200])}"
            )

        get_embeds            = self.llm.model.get_input_embeddings()
        part1_input_ids       = batch["part1_input_ids"]
        part1_attention_mask  = batch["part1_attention_mask"]

        part1_embeds  = get_embeds(part1_input_ids)   # [B, L1, D]
        target_embeds = get_embeds(target_input_ids)  # [B, L2, D]

        # Sequence: [user turn open] [context] [user turn close + assistant + answer]
        inputs_embeds = torch.cat([part1_embeds, h, target_embeds], dim=1)

        ctx_mask = torch.ones(
            h.size(0), h.size(1), device=h.device, dtype=target_attention_mask.dtype
        )
        attention_mask = torch.cat([part1_attention_mask, ctx_mask, target_attention_mask], dim=1)

        labels = batch["labels"]

        part1_ignore = torch.full(
            (labels.size(0), part1_input_ids.size(1)), -100, device=labels.device
        )
        ctx_ignore = torch.full(
            (labels.size(0), h.size(1)), -100, device=labels.device
        )
        labels = torch.cat([part1_ignore, ctx_ignore, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    # -----------------------------
    # training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        # Guard against degenerate batches (all-masked labels → NaN/inf loss)
        if not torch.isfinite(loss):
            return None

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/cumulative_flops",
            float(self.flops_per_step * (self.global_step + 1)),
            on_step=True,
            prog_bar=False,
        )
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms of all trainable components."""
        def _grad_norm(params):
            norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
            return torch.stack(norms).norm(2).item() if norms else 0.0

        mapper_norm = _grad_norm(self.mapper.parameters())
        self.log("train/mapper_grad_norm", mapper_norm, prog_bar=False, on_step=True)

        if self.embedder.trainable:
            emb_norm = _grad_norm(self.embedder.parameters())
            self.log("train/embedder_grad_norm", emb_norm, prog_bar=False, on_step=True)

        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False, on_step=True)

    # -----------------------------
    # validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        source_input_ids      = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]
        tasks     = batch.get("task",        ["default"] * source_input_ids.size(0))
        answers   = batch.get("answer",      [""] * source_input_ids.size(0))
        src_texts = batch.get("source_text", [""] * source_input_ids.size(0))
        questions = batch.get("question",    [""] * source_input_ids.size(0))
        get_embeds = self.llm.model.get_input_embeddings()

        # ─── Single embedder + mapper pass (shared for loss and generation) ───
        z = self.embedder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
        )
        h = self._wrap_context(self.mapper(z, source_attention_mask))  # [B, n_ctx+2, D]

        if self.soft_prompt is not None:
            sp = self.soft_prompt.expand(h.size(0), -1, -1).to(h.dtype)
            h = torch.cat([h, sp], dim=1)  # [B, n_ctx+2+p, D]

        # ─── Teacher-forcing loss (no second embedder call) ───────────────
        part1_input_ids       = batch["part1_input_ids"]
        part1_attention_mask  = batch["part1_attention_mask"]
        target_input_ids      = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]
        part1_embeds          = get_embeds(part1_input_ids)
        target_embeds         = get_embeds(target_input_ids)
        inputs_embeds         = torch.cat([part1_embeds, h, target_embeds], dim=1)
        ctx_mask              = torch.ones(
            h.size(0), h.size(1), device=h.device, dtype=target_attention_mask.dtype
        )
        attention_mask  = torch.cat([part1_attention_mask, ctx_mask, target_attention_mask], dim=1)
        labels          = batch["labels"]
        part1_ignore    = torch.full(
            (labels.size(0), part1_input_ids.size(1)), -100, device=labels.device
        )
        ctx_ignore      = torch.full(
            (labels.size(0), h.size(1)), -100, device=labels.device
        )
        labels_full     = torch.cat([part1_ignore, ctx_ignore, labels], dim=1)
        val_loss = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels_full
        ).loss
        self.log("val/loss", val_loss, prog_bar=True)

        if not self.val_generate:
            return val_loss

        # ─── Batched free generation (grouped by task) ────────────────────
        preds: list[str] = [""] * len(tasks)

        task_groups: dict[str, list[int]] = {}
        for i, t in enumerate(tasks):
            task_groups.setdefault(t, []).append(i)

        with torch.inference_mode():
            for task_name, indices in task_groups.items():
                h_group = h[indices]  # [G, n_ctx+2, D]

                # Build two-part prefix matching data_module.tokenize():
                # part1 = user turn open + message (before context tokens)
                # part2 = close user turn + open assistant + bypass (after context tokens)
                _BYPASS = "<think>\n\n</think>\n\n"

                if task_name == "qa":
                    part1_texts = [f"<|im_start|>user\n{questions[i]}" for i in indices]
                    part2_prefix = f"<|im_end|>\n<|im_start|>assistant\n{_BYPASS}"
                else:
                    part1_texts  = [f"<|im_start|>user\nReproduce the following text."] * len(indices)
                    part2_prefix = f"<|im_end|>\n<|im_start|>assistant\n{_BYPASS}<REPRODUCE>"

                part2_texts = [part2_prefix] * len(indices)

                self._dbg(
                    f"val gen task={task_name} n={len(indices)} "
                    f"part1[0]={repr(part1_texts[0][:60])}"
                )

                enc1 = self.llm_tokenizer(part1_texts, add_special_tokens=False,
                                          padding=True, return_tensors="pt")
                enc2 = self.llm_tokenizer(part2_texts, add_special_tokens=False,
                                          padding=True, return_tensors="pt")
                part1_ids  = enc1.input_ids.to(h.device)
                part1_mask = enc1.attention_mask.to(h.device)
                part2_ids  = enc2.input_ids.to(h.device)
                part2_mask = enc2.attention_mask.to(h.device)
                part1_embs = get_embeds(part1_ids)   # [G, L1, D]
                part2_embs = get_embeds(part2_ids)   # [G, L2, D]

                ctx_mask_gen = torch.ones(
                    len(indices), h_group.size(1), device=h.device, dtype=torch.long
                )
                gen_input = torch.cat([part1_embs, h_group, part2_embs], dim=1)
                gen_mask  = torch.cat([part1_mask, ctx_mask_gen, part2_mask], dim=1)

                out = self.llm.generate(
                    inputs_embeds=gen_input,
                    attention_mask=gen_mask,
                    max_new_tokens=64,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )

                for j, idx in enumerate(indices):
                    preds[idx] = self._strip_thinking(
                        self.llm_tokenizer.decode(out[j], skip_special_tokens=True)
                    )

                self._dbg(
                    f"val gen done pred[0]={repr(preds[indices[0]][:100])}"
                )

        for p, ans, src, q, task in zip(preds, answers, src_texts, questions, tasks):
            self.val_storage[task].append((p, ans, src, q))

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )

        if self.warmup_steps == 0:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # -----------------------------
    # epoch end
    # -----------------------------
    def on_fit_start(self):
        """Analytical FLOPs per training step."""
        self.flops_forward_embedder = self.embedder.forward_flops()
        self.flops_forward_mapper   = self.mapper.forward_flops()
        self.flops_forward_llm      = self.llm.forward_flops()

        trainable_flops = 0
        if self.embedder.trainable:
            trainable_flops += self.flops_forward_embedder
        if self.mapper.trainable:
            trainable_flops += self.flops_forward_mapper
        if self.llm.trainable:
            trainable_flops += self.flops_forward_llm
        # Soft-prompt: full LLM backward is needed to propagate gradient to the
        # learned vectors even when the base LLM weights are frozen.
        if self.soft_prompt is not None and not self.llm.trainable:
            trainable_flops += self.flops_forward_llm

        self.flops_per_step = (
            self.flops_forward_embedder +
            self.flops_forward_mapper +
            self.flops_forward_llm +
            2 * trainable_flops
        )

    def on_validation_epoch_end(self):
        for task, quads in self.val_storage.items():
            preds = [p   for p, _, _, _ in quads]
            refs  = [ans for _, ans, _, _ in quads]
            srcs  = [src for _, _, src, _ in quads]
            qs    = [q   for _, _, _, q in quads]

            # Compute all metrics at multiple word-level cutoffs so that
            # BLEU brevity penalty and ROUGE recall are computed fairly
            # (short predictions vs short reference prefixes, not full refs).
            # @64 ≈ max_new_tokens=64 tokens, the most meaningful cutoff.
            for cutoff in (64, 128, 256):
                p_cut = [self._trunc_words(p, cutoff) for p in preds]
                r_cut = [self._trunc_words(r, cutoff) for r in refs]

                rouge_scores = rouge_l(p_cut, r_cut)
                bleu_score   = bleu4(p_cut, r_cut)
                tok_acc      = self.compute_token_accuracy(p_cut, r_cut)
                prefix_len, prefix_ratio = self.compute_prefix_metrics(p_cut, r_cut, cutoff)

                self.log(f"val/{task}/rougeL@{cutoff}",      rouge_scores["rougeL"])
                self.log(f"val/{task}/bleu@{cutoff}",         bleu_score["bleu"])
                self.log(f"val/{task}/tok_acc@{cutoff}",      tok_acc)
                self.log(f"val/{task}/prefix_len@{cutoff}",   prefix_len)
                self.log(f"val/{task}/prefix_ratio@{cutoff}", prefix_ratio)

            self._log_val_table(task, preds, refs, srcs, qs)

        self.val_storage.clear()

    # -----------------------------
    # metrics / logging helpers
    # -----------------------------
    def compute_token_accuracy(self, preds, refs):
        correct = 0
        total = 0
        for p, r in zip(preds, refs):
            p_tokens = p.split()
            r_tokens = r.split()
            min_len = min(len(p_tokens), len(r_tokens))
            for i in range(min_len):
                if p_tokens[i] == r_tokens[i]:
                    correct += 1
            total += min_len
        return correct / total if total > 0 else 0.0

    def compute_prefix_metrics(self, preds, refs, cutoff: int):
        """
        prefix_len:   average number of whitespace-tokens matched consecutively
                      from position 0 until the first mismatch.

        prefix_ratio: prefix_len / min(cutoff, len(ref_tokens)).
                      Proportion of the reference (up to cutoff) reproduced
                      verbatim from the start.  E.g. 0.84 means 84% of the
                      first `cutoff` reference words were predicted in order.

        Both are averaged over the batch.
        """
        prefix_lens = []
        ratios      = []
        for p, r in zip(preds, refs):
            p_toks = p.split()
            r_toks = r.split()
            match_len = 0
            for pt, rt in zip(p_toks, r_toks):
                if pt == rt:
                    match_len += 1
                else:
                    break
            denom = min(cutoff, len(r_toks)) if r_toks else 1
            prefix_lens.append(match_len)
            ratios.append(match_len / denom)
        avg_prefix = sum(prefix_lens) / len(prefix_lens) if prefix_lens else 0.0
        avg_ratio  = sum(ratios)      / len(ratios)      if ratios      else 0.0
        return avg_prefix, avg_ratio

    def _log_val_table(self, task, preds, refs, srcs, questions):
        """Write a sample table to TensorBoard as markdown text."""
        logger = self.logger
        if isinstance(logger, list):
            logger = logger[0]
        if logger is None or not hasattr(logger, "experiment"):
            return
        exp = logger.experiment
        if not hasattr(exp, "add_text"):
            return

        show_question = task == "qa"
        if show_question:
            lines = [f"## {task} — epoch {self.current_epoch}\n"]
            lines.append("| # | Source (truncated) | Question | Prediction | Answer |")
            lines.append("|---|---|---|---|---|")
        else:
            lines = [f"## {task} — epoch {self.current_epoch}\n"]
            lines.append("| # | Source (truncated) | Prediction | Target |")
            lines.append("|---|---|---|---|")

        n = _VAL_LOG_EXAMPLES
        for i, (p, r, s, q) in enumerate(zip(preds[:n], refs[:n], srcs[:n], questions[:n])):
            def _esc(t): return t[:300].replace("|", "\\|").replace("\n", " ")
            s_disp = _esc(s[:200])
            p_disp = _esc(p)
            r_disp = _esc(r)
            if show_question:
                q_disp = _esc(q)
                lines.append(f"| {i + 1} | {s_disp} | {q_disp} | {p_disp} | {r_disp} |")
            else:
                lines.append(f"| {i + 1} | {s_disp} | {p_disp} | {r_disp} |")

        exp.add_text(f"val/{task}_samples", "\n".join(lines), self.global_step)
