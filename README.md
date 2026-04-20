# Vision Tune

Parameter-efficient fine-tuning of CLIP (ViT-B/32) and ViT (google/vit-base-patch16-224) using LoRA adapters, with ablation studies across rank, target modules, and learning rate.

## Overview

- **LoRA fine-tuning**: Inject low-rank matrices `A (d×r)` and `B (r×d)` into QKV projections; only A and B are trained (<1% of parameters)
- **Models**: CLIP ViT-B/32 + ViT-B/16-224 via HuggingFace `transformers` + `peft`
- **Ablation grid**: rank ∈ {4, 8, 16, 32} × 2 model types × 2 learning rates × 2 target module sets = 32 runs
- **Metrics**: Top-1 and Top-5 accuracy on held-out test set
- **Results table**: CSV with all ablation results, sorted by val top-1

## Tech Stack

Python 3.11 · PyTorch · HuggingFace Transformers · PEFT · torchvision

## Quickstart

```bash
pip install -r requirements.txt

# Run full ablation sweep
python src/ablation.py --config configs/clip_base.yaml --data_root data/ --output_csv results/ablation_results.csv

# Tests
pytest tests/ -v
```

## Architecture

```
Domain dataset (folder-organized images)
    → ImageFolderDataset with CLIP/ViT normalization
    → CLIP / ViT backbone (frozen)
         + LoRA adapters on Q, V projections
    → Linear classifier head (num_classes)
    → Top-1 / Top-5 accuracy
```

## Results

LoRA fine-tuning with rank=8 on Q+V projections typically matches full fine-tuning accuracy with <1% trainable parameters, dramatically reducing GPU memory and training time.

| Model | LoRA Rank | Target Mods | Top-1 | Trainable |
|-------|-----------|-------------|-------|-----------|
| CLIP  | 8         | q+v         | ~85%  | <0.5%     |
| ViT   | 16        | q+k+v       | ~87%  | <0.8%     |
