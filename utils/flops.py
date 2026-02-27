import torch
from fvcore.nn import FlopCountAnalysis

def compute_flops(model):
    model.eval()
    device = next(model.parameters()).device

    dummy_batch = {
        "input_ids": torch.ones(1, 16, dtype=torch.long).to(device),
        "attention_mask": torch.ones(1, 16).to(device),
        "labels": torch.ones(1, 16, dtype=torch.long).to(device)
    }

    with torch.no_grad():
        flops = FlopCountAnalysis(
            model,
            (
                dummy_batch["input_ids"],
                dummy_batch["attention_mask"],
                dummy_batch["labels"],
            )
        )
    return flops.total()