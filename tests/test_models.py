import os
import tempfile
import numpy as np
import torch
import pytest
from PIL import Image

from src.lora_config import make_lora_config, count_trainable_params
from src.dataset import ImageFolderDataset, get_clip_transforms, get_vit_transforms


def make_lora(r=8):
    return make_lora_config(r=r, alpha=16, target_modules=["q_proj", "v_proj"], dropout=0.1)


def test_count_trainable_params_clip():
    pytest.importorskip("peft")
    from src.model_clip import CLIPLoRAClassifier
    lora_cfg = make_lora(8)
    model = CLIPLoRAClassifier("openai/clip-vit-base-patch32", num_classes=5, lora_cfg=lora_cfg)
    trainable, total = count_trainable_params(model)
    assert trainable < total
    assert trainable / total < 0.05  # < 5% trainable


def test_clip_forward_shape():
    pytest.importorskip("peft")
    from src.model_clip import CLIPLoRAClassifier
    lora_cfg = make_lora(8)
    model = CLIPLoRAClassifier("openai/clip-vit-base-patch32", num_classes=5, lora_cfg=lora_cfg)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)
    assert not torch.isnan(out).any()


def test_dataset_from_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in ["cat", "dog", "bird"]:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(os.path.join(cls_dir, f"img_{i}.jpg"))

        ds = ImageFolderDataset(tmpdir, transform=get_clip_transforms(train=False))
        assert len(ds) == 15
        assert ds.num_classes == 3
        img_tensor, label = ds[0]
        assert img_tensor.shape == (3, 224, 224)
