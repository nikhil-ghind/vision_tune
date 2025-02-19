import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


def make_lora_config(r: int, alpha: int, target_modules: list, dropout: float) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )


def count_trainable_params(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
