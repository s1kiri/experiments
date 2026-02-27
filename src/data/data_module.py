# src/data/text_datamodule.py

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pytorch_lightning as pl
from typing import Optional, Dict, Any, List


class TextDataset(Dataset):
    """
    Thin wrapper around column-typed datasets (HF datasets, pandas, etc).
    Expects dict-like access: dataset[i]["column_name"].
    """

    def __init__(self, dataset, task_filter=None):
        if task_filter:
           self.dataset = dataset.filter(lambda x: x['task'] == task_filter)
        else:
            self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        
        return {
            "source_text": row.get("source_text"),
            "target_text": row.get("target_text"),

            "tokenized_source_text": row.get("tokenized_source_text"),
            "tokenized_target_text": row.get("tokenized_target_text"),

            "am_source_text": row.get("am_source_text"),
            "am_target_text": row.get("am_target_text"),

            "task": row.get("task"),
            "id": row.get("id", idx),
        }


def text_collate_fn(batch, src_pad_id=0, tgt_pad_id=0):
    """
    Pads source and target sequences separately.
    Prepares labels for autoregressive LLM training.
    """

    from torch.nn.utils.rnn import pad_sequence

    source_ids = [
        torch.tensor(sample["tokenized_source_text"], dtype=torch.long)
        for sample in batch
    ]

    source_input_ids = pad_sequence(
        source_ids,
        batch_first=True,
        padding_value=src_pad_id
    )

    source_attention_mask = (source_input_ids != src_pad_id).long()

    target_ids = [
        torch.tensor(sample["tokenized_target_text"], dtype=torch.long)
        for sample in batch
    ]

    target_input_ids = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=tgt_pad_id
    )

    target_attention_mask = (target_input_ids != tgt_pad_id).long()

    labels = target_input_ids.clone()

    labels[target_input_ids == tgt_pad_id] = -100

    tasks = [sample.get("task", "default") for sample in batch]
    ids = [sample.get("id", i) for i, sample in enumerate(batch)]

    return {
        # embedder inputs
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,

        # llm inputs
        "target_input_ids": target_input_ids,
        "target_attention_mask": target_attention_mask,

        # training
        "labels": labels,
        "task": tasks,
        "id": ids,

        # raw texts for validation logging (lists of strings, not tensors)
        "source_text": [sample.get("source_text", "") for sample in batch],
        "target_text": [sample.get("target_text", "") for sample in batch],
    }



class TextDataModule(pl.LightningDataModule):
    """
    End-to-end text DataModule for narrative / QA / RAG-like experiments.
    """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        emb_tok,
        llm_tok,
        emb_max_length: int = 512,
        llm_max_length: int = 256,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.emb_tok = emb_tok
        self.llm_tok = llm_tok
        self.emb_max_length = emb_max_length
        self.llm_max_length = llm_max_length

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        train_dataset = self.prepare_dataset(train_dataset)
        val_dataset = self.prepare_dataset(val_dataset)

        self.train_dataset = TextDataset(train_dataset)
        self.val_dataset = TextDataset(val_dataset)

    def _collate_fn(self):
        return partial(
            text_collate_fn,
            src_pad_id=self.emb_tok.pad_token_id or 0,
            tgt_pad_id=self.llm_tok.pad_token_id or 0,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            drop_last=False,
        )
    

    def prepare_dataset(self, dataset):
        tokenized = dataset.map(
            self.tokenize,
            # remove_columns=dataset["train"].column_names,
            batched=False
        )
        return tokenized

    def tokenize(self, example):
        src = self.emb_tok(
            example["source_text"],
            truncation=True,
            max_length=self.emb_max_length,
        )

        tgt = self.llm_tok(
            example["target_text"],
            truncation=True,
            max_length=self.llm_max_length,
        )
    
        return {
            "source_text": example["source_text"],
            "target_text": example["target_text"],

            "tokenized_source_text": src["input_ids"],
            "tokenized_target_text": tgt["input_ids"],
            
            "am_source_text": src["attention_mask"],
            "am_target_text": tgt["attention_mask"],

            "task": example["task"],
            "id": example["id"],
        }
