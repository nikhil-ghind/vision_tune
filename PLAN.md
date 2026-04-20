# Vision Tune

## Project Overview

Fine-tune CLIP (ViT-B/32) and a standalone ViT (google/vit-base-patch16-224) on a domain-specific image classification dataset using LoRA (Low-Rank Adaptation) adapters to achieve parameter-efficient fine-tuning. Ablation studies are run across LoRA rank, adapter placement, and learning rate, measuring top-1 and top-5 accuracy on a held-out test set. The project demonstrates that LoRA fine-tuning with <1% trainable parameters can approach or match full fine-tuning accuracy.

Key goals:
- Adapt CLIP and ViT to a narrow domain (e.g. medical imaging, satellite patches, fine-grained species) without training all parameters.
- Systematically ablate LoRA rank (4, 8, 16, 32), target modules, and learning rate.
- Track all runs with a structured results table comparing top-1/top-5 accuracy and trainable parameter count.
- Keep total fine-tuning time under 2 hours on a single A100/V100 for the full sweep.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Foundation models | CLIP (openai/clip-vit-base-patch32), ViT (google/vit-base-patch16-224) via HuggingFace `transformers 4.39` |
| LoRA | `peft 0.10` (HuggingFace PEFT library) |
| Training | PyTorch 2.2, `torch.cuda.amp` for mixed precision |
| Data | torchvision 0.17, Pillow 10, pandas 2.2 |
| Evaluation | scikit-learn 1.4, numpy 1.26 |
| Experiment logging | Python `logging`, CSV results table |
| Testing | pytest 8.1 |
| Environment | Python 3.11, CUDA 12.1 |

---

## Architecture Overview

```
[Domain Dataset (images + labels)]
        |
        v
[DataLoader with augmentation transforms]
        |
   +---------+---------+
   |                   |
   v                   v
[CLIP ViT encoder]  [ViT encoder]
+ LoRA adapters     + LoRA adapters
on QKV projections  on QKV projections
   |                   |
   v                   v
[Linear classifier head (frozen or learned)]
   |                   |
   v                   v
[Top-1 / Top-5 accuracy evaluation]
        |
        v
[Ablation results CSV: rank x model x lr → metrics]
```

LoRA inserts low-rank matrices `A (d x r)` and `B (r x d)` alongside each frozen weight matrix `W`. Only `A` and `B` are trained. The classifier head (linear layer mapping final embedding to `num_classes`) is always trained from scratch.

---

## Phase 1: Project Scaffolding and Environment

**Goal**: Set up the project structure, install dependencies, and prepare the dataset.

### Tasks

1. Create the directory layout:
   ```
   visionFoundationModelFineTuning/
   ├── src/
   │   ├── __init__.py
   │   ├── dataset.py
   │   ├── model_clip.py
   │   ├── model_vit.py
   │   ├── lora_config.py
   │   ├── train.py
   │   ├── evaluate.py
   │   └── ablation.py
   ├── configs/
   │   ├── clip_base.yaml
   │   └── vit_base.yaml
   ├── data/
   ├── results/
   │   └── ablation_results.csv
   ├── checkpoints/
   ├── requirements.txt
   └── tests/
       └── test_models.py
   ```

2. Create `requirements.txt`:
   ```
   torch==2.2.2
   torchvision==0.17.2
   transformers==4.39.3
   peft==0.10.0
   Pillow==10.2.0
   numpy==1.26.4
   scikit-learn==1.4.1
   pandas==2.2.1
   tqdm==4.66.2
   pyyaml==6.0.1
   pytest==8.1.1
   ```

3. Install: `pip install -r requirements.txt`.

4. Create `configs/clip_base.yaml`:
   ```yaml
   model:
     name: openai/clip-vit-base-patch32
     type: clip
     num_classes: <N>
   lora:
     r: 8
     lora_alpha: 16
     target_modules: ["q_proj", "v_proj"]
     lora_dropout: 0.1
     bias: "none"
   training:
     epochs: 20
     batch_size: 64
     lr: 1e-4
     weight_decay: 1e-4
     warmup_ratio: 0.1
     fp16: true
   data:
     train_dir: data/train
     val_dir: data/val
     test_dir: data/test
     image_size: 224
   ```

5. Create `configs/vit_base.yaml` mirroring the above but with `model.name: google/vit-base-patch16-224` and `model.type: vit`.

---

## Phase 2: Dataset Module

**Goal**: Build a flexible dataset class supporting any folder-organized image classification dataset.

### Tasks

1. Create `src/dataset.py`:
   - `class ImageFolderDataset(torch.utils.data.Dataset)`:
     - `__init__(self, root_dir: str, transform=None)`: walks directory, collects `(path, label_int)` pairs. Infers `class_to_idx` from sorted subdirectory names.
     - `__len__`, `__getitem__`: loads image with PIL, applies transform, returns `(tensor, label)`.
     - Property `num_classes: int`.
     - Property `class_names: list[str]`.
   - `get_clip_transforms(train: bool, image_size: int = 224) -> transforms.Compose`:
     - Train: `RandomResizedCrop(image_size, scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `ColorJitter(0.3, 0.3, 0.3)`, `ToTensor()`, `Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])` (CLIP normalization).
     - Val/Test: `Resize(image_size + 32)`, `CenterCrop(image_size)`, same normalize.
   - `get_vit_transforms(train: bool, image_size: int = 224) -> transforms.Compose`: same structure but with ImageNet normalization `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.

2. Write a test in `tests/test_models.py` that instantiates `ImageFolderDataset` on a 3-class, 15-image synthetic dataset and verifies `len`, `num_classes`, and returned tensor shapes.

---

## Phase 3: Model Construction with LoRA

**Goal**: Wrap CLIP and ViT with PEFT LoRA adapters and attach classification heads.

### Tasks

1. Create `src/lora_config.py`:
   - `def make_lora_config(r: int, alpha: int, target_modules: list[str], dropout: float) -> LoraConfig`:
     - Returns `peft.LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias="none")`.
   - `def count_trainable_params(model: nn.Module) -> tuple[int, int]`: returns `(trainable, total)` parameter counts.

2. Create `src/model_clip.py`:
   - `class CLIPLoRAClassifier(nn.Module)`:
     - `__init__(self, model_name: str, num_classes: int, lora_cfg: LoraConfig)`:
       - Load `CLIPModel.from_pretrained(model_name)`.
       - Apply LoRA: `self.clip = get_peft_model(clip_model, lora_cfg)`.
       - Freeze all params except LoRA: `self.clip.print_trainable_parameters()` (this is handled by PEFT).
       - Add `self.classifier = nn.Linear(512, num_classes)` (CLIP ViT-B/32 outputs 512-d).
     - `forward(self, pixel_values: Tensor) -> Tensor`: call `self.clip.get_image_features(pixel_values=pixel_values)`, L2-normalize the features, pass through `self.classifier`, return logits.

3. Create `src/model_vit.py`:
   - `class ViTLoRAClassifier(nn.Module)`:
     - `__init__(self, model_name: str, num_classes: int, lora_cfg: LoraConfig)`:
       - Load `ViTModel.from_pretrained(model_name)`.
       - Apply LoRA: `self.vit = get_peft_model(vit_model, lora_cfg)`.
       - Add `self.classifier = nn.Linear(768, num_classes)` (ViT-B outputs 768-d CLS token).
     - `forward(self, pixel_values: Tensor) -> Tensor`: call `self.vit(pixel_values=pixel_values)`, extract `outputs.last_hidden_state[:, 0, :]` (CLS token), pass through `self.classifier`, return logits.

4. Add tests in `tests/test_models.py`:
   - Instantiate `CLIPLoRAClassifier` with `num_classes=5`, `r=8`. Assert trainable params < 1% of total.
   - Instantiate `ViTLoRAClassifier` with `num_classes=5`, `r=8`. Assert trainable params < 1% of total.
   - Forward pass with `(2, 3, 224, 224)` input. Assert output shape is `(2, 5)`.

---

## Phase 4: Training Loop

**Goal**: Implement the training and evaluation loops used by all ablation runs.

### Tasks

1. Create `src/train.py`:
   - `train_one_epoch(model, loader, optimizer, criterion, scaler, device) -> float`: standard mixed-precision training loop returning mean loss.
   - `evaluate(model, loader, device) -> dict`:
     - Runs model in eval mode.
     - Collects all logits and labels.
     - Computes `top1_accuracy = (preds == labels).float().mean()`.
     - Computes top-5 accuracy: `torch.topk(logits, k=5, dim=1)` → check if true label in top-5 predictions.
     - Returns `{"top1": float, "top5": float}`.
   - `train_model(config: dict, model: nn.Module, train_loader, val_loader, device) -> dict`:
     - Optimizer: `AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])`.
     - Scheduler: `get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=..., num_training_steps=total_steps)`.
     - Scaler: `torch.cuda.amp.GradScaler(enabled=config["fp16"])`.
     - Loop for `epochs`: `train_one_epoch`, `evaluate` on val, track best val top-1, save best checkpoint.
     - Returns final `{"best_val_top1": float, "best_val_top5": float}`.

---

## Phase 5: Ablation Study Runner

**Goal**: Systematically sweep LoRA hyperparameters and record results.

### Tasks

1. Create `src/ablation.py`:
   - Define `ABLATION_GRID`:
     ```python
     ABLATION_GRID = {
         "model_type": ["clip", "vit"],
         "lora_rank": [4, 8, 16, 32],
         "lr": [1e-4, 5e-5],
         "target_modules": [
             ["q_proj", "v_proj"],
             ["q_proj", "k_proj", "v_proj"],
         ],
     }
     ```
   - `def run_ablation(base_config: dict, data_root: str, output_csv: str, device: str)`:
     - Enumerate all combinations with `itertools.product`.
     - For each combination:
       - Build `LoraConfig` via `make_lora_config`.
       - Build model (`CLIPLoRAClassifier` or `ViTLoRAClassifier`).
       - Build datasets and dataloaders.
       - Call `train_model` with overridden `lr`.
       - Record `{model_type, lora_rank, lr, target_modules, top1, top5, trainable_params, total_params}` as a row.
     - Write all rows to `output_csv` via pandas DataFrame.
   - `if __name__ == "__main__"`: parse `--config`, `--data_root`, `--output_csv` from argparse, call `run_ablation`.

2. Run ablation:
   ```
   python src/ablation.py --config configs/clip_base.yaml --data_root data/ --output_csv results/ablation_results.csv
   ```

3. After the run, print the top-3 configs by val top-1 accuracy and log the trainable parameter count for each.

---

## Phase 6: Final Evaluation and Reporting

**Goal**: Evaluate the best configuration on the held-out test set and produce a final results summary.

### Tasks

1. Create `src/evaluate.py` (standalone script):
   - Load the best checkpoint identified from `ablation_results.csv` (highest val top-1).
   - Reconstruct the model with the same config (model type, LoRA rank, target modules).
   - Load `peft` adapter weights from checkpoint.
   - Run `evaluate(model, test_loader, device)`.
   - Print and save `results/final_test_results.txt` with: model name, LoRA rank, trainable params, test top-1, test top-5.

2. Add `tests/test_models.py` integration test:
   - Instantiate the best config model.
   - Run a forward pass on a batch of 4 images.
   - Assert output shape, no NaN in logits.

3. Run the full pytest suite:
   ```
   pytest tests/ -v
   ```
   All tests must pass.

4. Summarize findings: record which model type and LoRA rank provides the best top-1/top-5 tradeoff, note trainable parameter percentage.
