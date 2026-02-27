# src/data/data_module.py

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pytorch_lightning as pl
from typing import Dict, Any

from src.data.schema import TASKS


class TextDataset(Dataset):
    """
    Thin wrapper around column-typed datasets (HF datasets, pandas, etc).
    Expects dict-like access: dataset[i]["column_name"].
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        return {
            "source_text":           row.get("source_text", ""),
            "question":              row.get("question", ""),
            "answer":                row.get("answer", ""),
            "tokenized_source_text": row.get("tokenized_source_text"),
            "tokenized_target_text": row.get("tokenized_target_text"),
            "n_label_mask":          row.get("n_label_mask", 0),
            "task":                  row.get("task"),
            "id":                    row.get("id", idx),
        }


def text_collate_fn(batch, src_pad_id=0, tgt_pad_id=0):
    """
    Pads source and target sequences separately.
    Prepares labels for autoregressive LLM training.

    For QA, the question-prefix tokens are masked with -100 so the model
    is trained only on the answer tokens.
    """
    from torch.nn.utils.rnn import pad_sequence

    source_ids = [
        torch.tensor(sample["tokenized_source_text"], dtype=torch.long)
        for sample in batch
    ]
    source_input_ids = pad_sequence(source_ids, batch_first=True, padding_value=src_pad_id)
    source_attention_mask = (source_input_ids != src_pad_id).long()

    target_ids = [
        torch.tensor(sample["tokenized_target_text"], dtype=torch.long)
        for sample in batch
    ]
    target_input_ids = pad_sequence(target_ids, batch_first=True, padding_value=tgt_pad_id)
    target_attention_mask = (target_input_ids != tgt_pad_id).long()

    labels = target_input_ids.clone()
    labels[target_input_ids == tgt_pad_id] = -100

    # Mask the task-prefix tokens per example (question for QA, <REPRODUCE> for narrative)
    for i, sample in enumerate(batch):
        n_mask = sample.get("n_label_mask", 0)
        if n_mask > 0:
            labels[i, :n_mask] = -100

    tasks = [sample.get("task", "default") for sample in batch]
    ids   = [sample.get("id", i) for i, sample in enumerate(batch)]

    return {
        # embedder inputs
        "source_input_ids":      source_input_ids,
        "source_attention_mask": source_attention_mask,

        # llm inputs
        "target_input_ids":      target_input_ids,
        "target_attention_mask": target_attention_mask,

        # training
        "labels": labels,
        "task":   tasks,
        "id":     ids,

        # raw texts for validation logging (lists of strings)
        "source_text": [sample.get("source_text", "") for sample in batch],
        "question":    [sample.get("question", "")    for sample in batch],
        "answer":      [sample.get("answer", "")      for sample in batch],
    }


class TextDataModule(pl.LightningDataModule):
    """
    End-to-end text DataModule for narrative / QA / RAG-like experiments.

    Expected dataset columns: id, split, task, source_text, question, answer
    (see src/data/schema.py for the full specification).
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

        self.train_dataset = TextDataset(self.prepare_dataset(train_dataset))
        self.val_dataset   = TextDataset(self.prepare_dataset(val_dataset))

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
        return dataset.map(self.tokenize, batched=False)

    def tokenize(self, example):
        task = example["task"]

        src_enc = self.emb_tok(
            example["source_text"],
            truncation=True,
            max_length=self.emb_max_length,
            add_special_tokens=True,
        )

        if task == "narrative":
            prefix_text = "<REPRODUCE>"
            answer_text = example["answer"]
        elif task == "qa":
            prefix_text = f"<QA> {example['question']} <ANSWER>"
            answer_text = example["answer"]
        else:
            raise ValueError(f"Unknown task: {task!r}. Expected one of {TASKS}.")

        prefix_ids = self.llm_tok(
            prefix_text,
            add_special_tokens=False,
        ).input_ids

        answer_ids = self.llm_tok(
            answer_text,
            truncation=True,
            max_length=self.llm_max_length - len(prefix_ids),
            add_special_tokens=False,
        ).input_ids

        return {
            "source_text":           example["source_text"],
            "question":              example.get("question", ""),
            "answer":                example["answer"],
            "task":                  task,
            "id":                    example["id"],
            "tokenized_source_text": src_enc["input_ids"],
            "am_source_text":        src_enc["attention_mask"],
            "tokenized_target_text": prefix_ids + answer_ids,
            "n_label_mask":          len(prefix_ids),
        }
