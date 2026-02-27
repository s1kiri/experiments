import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
import torch
from datasets import load_from_disk


from src.models.embedder import HFEmbedder
from src.models.mapper import BaseMapper
from src.models.llm import HFLLM
from src.models.lightning_module import MapperLLMModule
from src.data.data_module import TextDataModule


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str, train_ds, val_ds):
    cfg = load_config(config_path)

    pl.seed_everything(cfg["experiment"]["seed"])

    # --------------------
    # tokenizer
    # --------------------
    emb_tok = AutoTokenizer.from_pretrained(cfg["models"]["embedder"]["name"])
    llm_tok = AutoTokenizer.from_pretrained(cfg["models"]["llm"]["name"])
    llm_tok.pad_token = llm_tok.eos_token

    # --------------------
    # models
    # --------------------
    embedder_cfg = cfg["models"]["embedder"]
    embedder = HFEmbedder(
        model_name=embedder_cfg["name"],
        trainable=embedder_cfg["trainable"],
        max_length=embedder_cfg["max_length"],
    )

    mapper_cfg = cfg["models"]["mapper"]
    mapper = BaseMapper(**mapper_cfg["params"])
    mapper.trainable = mapper_cfg["trainable"]

    llm_cfg = cfg["models"]["llm"]
    llm = HFLLM(
        model_name=llm_cfg["name"],
        trainable=llm_cfg["trainable"],
        max_length=llm_cfg["max_length"],
    )

    # --------------------
    # lightning module
    # --------------------
    model = MapperLLMModule(
        embedder=embedder,
        mapper=mapper,
        llm=llm,
        llm_tokenizer=llm_tok,
        lr=cfg["training"]["lr"],
        target_metric=cfg["compute"]["target_metric"],
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
    # trainer
    # --------------------
    use_cuda = torch.cuda.is_available()
    # fp16 is only supported on GPU; fall back to fp32 on CPU (Mac dev)
    precision = cfg["training"]["precision"] if use_cuda else 32

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        logger=logger,
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        precision=precision,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    dataset = load_from_disk("data/narrativeqa_dataset")

    train_ds = dataset["train"].select(range(500))
    val_ds = dataset["validation"].select(range(50))
    main("configs/base_config.yaml", train_ds, val_ds)
