import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import get_peft_model, LoraConfig


class CLIPLoRAClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, lora_cfg: LoraConfig):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(model_name)
        self.clip = get_peft_model(clip_model, lora_cfg)
        embed_dim = clip_model.config.projection_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.clip.get_image_features(pixel_values=pixel_values)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return self.classifier(features)
