import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
import torch
from datasets import load_from_disk


from src.models.embedder import HFEmbedder
from src.models.mapper import *
from src.models.llm import HFLLM
from src.models.lightning_module import MapperLLMModule
from src.data.data_module import TextDataModule
from src.data.schema import SPECIAL_TOKENS


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model(name, cache_dir) -> str:
    """Return a local path if the model was downloaded, otherwise return the HF hub name.

    download_models.py saves each model to <cache_dir>/<name with / replaced by -->.
    If that directory exists we load from it (fully offline); otherwise HF downloads it.
    """
    if cache_dir:
        local = os.path.join(cache_dir, name.replace("/", "--"))
        if os.path.isdir(local):
            return local
    return name


def main(config_path: str, train_ds, val_ds):
    cfg = load_config(config_path)

    pl.seed_everything(cfg["experiment"]["seed"])

    cache_dir = cfg["models"].get("cache_dir") or None

    # --------------------
    # tokenizer
    # --------------------
    emb_tok = AutoTokenizer.from_pretrained(
        resolve_model(cfg["models"]["embedder"]["name"], cache_dir)
    )
    # Ensure emb_tok has a pad token so pad_sequence uses a valid fill value.
    # The actual mask is carried via am_source_text, not inferred from the pad ID.
    if emb_tok.pad_token_id is None:
        emb_tok.pad_token_id = emb_tok.eos_token_id
    llm_tok = AutoTokenizer.from_pretrained(
        resolve_model(cfg["models"]["llm"]["name"], cache_dir)
    )
    llm_tok.pad_token = llm_tok.eos_token
    llm_tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # --------------------
    # models
    # --------------------
    embedder_cfg = cfg["models"]["embedder"]
    embedder = HFEmbedder(
        model_name=resolve_model(embedder_cfg["name"], cache_dir),
        trainable=embedder_cfg["trainable"],
        max_length=embedder_cfg["max_length"],
    )

    mapper_cfg = cfg["models"]["mapper"]
    mapper_class = globals()[mapper_cfg["class"]]
    mapper = mapper_class(**mapper_cfg["params"])
    mapper.trainable = mapper_cfg["trainable"]

    llm_cfg = cfg["models"]["llm"]
    llm = HFLLM(
        model_name=resolve_model(llm_cfg["name"], cache_dir),
        trainable=llm_cfg["trainable"],
        max_length=llm_cfg["max_length"],
    )
    # Resize token embeddings to cover the newly added special tokens.
    # New rows are frozen along with the rest of the LLM when trainable=False.
    llm.model.resize_token_embeddings(len(llm_tok))

    # Initialize new special-token embeddings to the mean of existing embeddings.
    # Random init (the default) puts them OOD for the LLM, causing garbage generation.
    with torch.no_grad():
        emb = llm.model.get_input_embeddings()
        old_vocab = emb.weight.size(0) - len(SPECIAL_TOKENS)
        mean_emb  = emb.weight[:old_vocab].mean(0, keepdim=True)
        emb.weight[old_vocab:] = mean_emb.expand(len(SPECIAL_TOKENS), -1)

    # Enable gradient checkpointing for a trainable embedder to save activation memory.
    # Only worthwhile on CUDA where ~7-9 GB of activations would OOM the GPU.
    # On CPU memory is abundant and recomputing activations is slower, not faster.
    device_str = cfg["compute"].get("device", "cpu")
    if embedder_cfg["trainable"] and device_str.startswith("cuda"):
        embedder.model.gradient_checkpointing_enable()

    # --------------------
    # lightning module
    # --------------------
    model = MapperLLMModule(
        embedder=embedder,
        mapper=mapper,
        llm=llm,
        llm_tokenizer=llm_tok,
        lr=cfg["training"]["lr"],
        warmup_steps=cfg["training"].get("warmup_steps", 0),
        target_metric=cfg["compute"]["target_metric"],
        val_generate=cfg.get("val_generate", True),
        debug=cfg.get("debug", False),
    )

    # --------------------
    # data
    # --------------------
    datamodule = TextDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        emb_tok=emb_tok,
        llm_tok=llm_tok,
        emb_max_length=embedder_cfg["max_length"],
        llm_max_length=llm_cfg["max_length"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------
    # logging
    # --------------------
    logger = TensorBoardLogger(
        save_dir=cfg["experiment"]["output_dir"],
        name=cfg["experiment"]["name"],
    )

    # --------------------
    # checkpointing
    # --------------------
    ckpt_cfg = cfg.get("checkpointing", {})
    callbacks = []
    if ckpt_cfg.get("enabled", True):
        callbacks.append(ModelCheckpoint(
            monitor=ckpt_cfg.get("monitor", "val/loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_last=ckpt_cfg.get("save_last", True),
            filename="epoch={epoch:02d}-step={step:08d}",
            auto_insert_metric_name=False,
            # dirpath defaults to <logger.log_dir>/checkpoints/
        ))

    # --------------------
    # trainer
    # --------------------
    if device_str.startswith("cuda"):
        accelerator = "gpu"
        # "cuda:0" → devices=[0]; "cuda" → devices=[0]
        device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
        devices = [device_idx]
        precision = cfg["training"]["precision"]
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32  # fp16 is unsupported on CPU

    # Mid-epoch validation: GPU uses num_vals_per_epoch, CPU always validates once
    # per epoch (generate() is prohibitively slow on CPU for frequent checks).
    num_vals_per_epoch = cfg["training"].get("num_vals_per_epoch", 1)
    if device_str.startswith("cuda"):
        val_check_interval = 1.0 / num_vals_per_epoch
    else:
        val_check_interval = 1.0

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        val_check_interval=val_check_interval,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    # Expected schema: id, split, task, source_text, question, answer
    # See src/data/schema.py
    # config_path = "configs/base_config.yaml"
    config_path = "configs/qwen_mlp_config.yaml"
    cfg = load_config(config_path)

    dataset  = load_from_disk("data/unified_dataset")
    seed     = cfg["experiment"]["seed"]
    n_train  = cfg["data"].get("n_samples_train") or None   # None = use all remaining
    n_val    = cfg["data"].get("n_samples_val", 1000)

    # Carve out val first (no overlap with train), then take n_samples_train from rest.
    shuffled     = dataset["train"].shuffle(seed=seed)
    n_val_actual = min(n_val, len(shuffled))
    val_ds       = shuffled.select(range(n_val_actual))
    remaining    = shuffled.select(range(n_val_actual, len(shuffled)))
    n_train_actual = min(n_train or len(remaining), len(remaining))
    train_ds     = remaining.select(range(n_train_actual))

    main(config_path, train_ds, val_ds)
