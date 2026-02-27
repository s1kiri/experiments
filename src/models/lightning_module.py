import torch
import evaluate
from collections import defaultdict
import pytorch_lightning as pl
from src.models.embedder import HFEmbedder
from src.models.llm import HFLLM
from src.models.mapper import BaseMapper

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
        target_metric: None = None,
    ):
        super().__init__()

        self.embedder = embedder
        self.mapper = mapper
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.lr = lr

        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

        # stores (pred, target, source_text) triples per task across validation steps
        self.val_storage = defaultdict(list)

        self.target_metric = target_metric
        self.reached_target = False
        self.compute_to_quality = None

        self.save_hyperparameters(ignore=["embedder", "mapper", "llm", "llm_tokenizer"])

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, batch):
        source_input_ids = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]

        z = self.embedder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
        )
        # z: [B, S, D_e]

        h = self.mapper(z)
        # h: [B, S, D_llm]

        target_input_ids = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]

        target_embeds = self.llm.model.get_input_embeddings()(target_input_ids)

        inputs_embeds = torch.cat([h, target_embeds], dim=1)

        source_mask = torch.ones(
            h.size(0),
            h.size(1),
            device=h.device,
            dtype=target_attention_mask.dtype
        )

        attention_mask = torch.cat(
            [source_mask, target_attention_mask],
            dim=1
        )

        labels = batch["labels"]

        prefix_ignore = torch.full(
            (labels.size(0), h.size(1)),
            -100,
            device=labels.device
        )

        labels = torch.cat([prefix_ignore, labels], dim=1)

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

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm of the mapper (only trainable component)."""
        norms = [
            p.grad.detach().norm(2)
            for p in self.mapper.parameters()
            if p.grad is not None
        ]
        if norms:
            total_norm = torch.stack(norms).norm(2).item()
            self.log("train/mapper_grad_norm", total_norm, prog_bar=False, on_step=True)

        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False, on_step=True)

    # -----------------------------
    # validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):

        # ===== teacher-forcing loss =====
        outputs = self(batch)
        val_loss = outputs.loss
        self.log("val/loss", val_loss, prog_bar=True)

        # ===== free generation =====
        source_input_ids = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]

        with torch.no_grad():
            z = self.embedder(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask
            )

            h = self.mapper(z)

            generated = self.llm.generate(
                inputs_embeds=h,
                max_new_tokens=64,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )

        preds = self.llm_tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        targets = self.llm_tokenizer.batch_decode(
            batch["target_input_ids"],
            skip_special_tokens=True
        )

        tasks = batch.get("task", ["default"] * len(preds))
        source_texts = batch.get("source_text", [""] * len(preds))

        for p, t, src, task in zip(preds, targets, source_texts, tasks):
            self.val_storage[task].append((p, t, src))

        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )

    # -----------------------------
    # epoch end
    # -----------------------------
    def on_fit_start(self):
        """Analytical FLOPs per training step."""
        self.flops_forward_embedder = self.embedder.forward_flops()
        self.flops_forward_mapper = self.mapper.forward_flops()
        self.flops_forward_llm = self.llm.forward_flops()

        trainable_flops = 0
        if self.embedder.trainable:
            trainable_flops += self.flops_forward_embedder
        if self.mapper.trainable:
            trainable_flops += self.flops_forward_mapper
        if self.llm.trainable:
            trainable_flops += self.flops_forward_llm

        self.flops_per_step = (
            self.flops_forward_embedder +
            self.flops_forward_mapper +
            self.flops_forward_llm +
            2 * trainable_flops
        )

    def on_validation_epoch_end(self):
        for task, triplets in self.val_storage.items():
            preds = [p for p, _, _ in triplets]
            refs  = [t for _, t, _ in triplets]
            srcs  = [s for _, _, s in triplets]

            rouge_scores = self.rouge.compute(
                predictions=preds,
                references=refs
            )

            bleu_score = self.bleu.compute(
                predictions=preds,
                references=[[r] for r in refs]
            )

            token_acc = self.compute_token_accuracy(preds, refs)

            self.log(f"val/{task}/rougeL",    rouge_scores["rougeL"])
            self.log(f"val/{task}/bleu",       bleu_score["bleu"])
            self.log(f"val/{task}/token_acc",  token_acc)

            self._log_val_table(task, preds, refs, srcs)

        self.val_storage.clear()

    # -----------------------------
    # helpers
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

    def _log_val_table(self, task, preds, refs, srcs):
        """Write a sample table to TensorBoard as markdown text."""
        logger = self.logger
        if isinstance(logger, list):
            logger = logger[0]
        if logger is None or not hasattr(logger, "experiment"):
            return
        exp = logger.experiment
        if not hasattr(exp, "add_text"):
            return

        lines = [f"## {task} — epoch {self.current_epoch}\n"]
        lines.append("| # | Source (truncated) | Prediction | Target |")
        lines.append("|---|---|---|---|")

        for i, (p, r, s) in enumerate(zip(preds[:_VAL_LOG_EXAMPLES], refs[:_VAL_LOG_EXAMPLES], srcs[:_VAL_LOG_EXAMPLES])):
            # truncate long source; escape markdown pipe chars
            s_disp = s[:200].replace("|", "\\|").replace("\n", " ")
            p_disp = p.replace("|", "\\|").replace("\n", " ")
            r_disp = r.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i + 1} | {s_disp} | {p_disp} | {r_disp} |")

        exp.add_text(f"val/{task}_samples", "\n".join(lines), self.global_step)
