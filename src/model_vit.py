import torch
import torch.nn as nn
from transformers import ViTModel
from peft import get_peft_model, LoraConfig


class ViTLoRAClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, lora_cfg: LoraConfig):
        super().__init__()
        vit_model = ViTModel.from_pretrained(model_name)
        self.vit = get_peft_model(vit_model, lora_cfg)
        hidden_size = vit_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)
