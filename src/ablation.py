import argparse
import itertools
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.lora_config import make_lora_config, count_trainable_params
from src.model_clip import CLIPLoRAClassifier
from src.model_vit import ViTLoRAClassifier
from src.dataset import ImageFolderDataset, get_clip_transforms, get_vit_transforms
from src.train import train_model

ABLATION_GRID = {
    "model_type": ["clip", "vit"],
    "lora_rank": [4, 8, 16, 32],
    "lr": [1e-4, 5e-5],
    "target_modules": [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj"],
    ],
}

MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224",
}


def run_ablation(base_config: dict, data_root: str, output_csv: str, device: str):
    keys = list(ABLATION_GRID.keys())
    values = list(ABLATION_GRID.values())
    rows = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"\nRun: {params}")

        model_type = params["model_type"]
        lora_rank = params["lora_rank"]
        lr = params["lr"]
        target_mods = params["target_modules"]

        get_tf = get_clip_transforms if model_type == "clip" else get_vit_transforms
        train_ds = ImageFolderDataset(f"{data_root}/train", transform=get_tf(train=True))
        val_ds = ImageFolderDataset(f"{data_root}/val", transform=get_tf(train=False))
        num_classes = train_ds.num_classes

        train_loader = DataLoader(train_ds, batch_size=base_config["batch_size"], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=base_config["batch_size"], num_workers=2)

        lora_cfg = make_lora_config(lora_rank, lora_rank * 2, target_mods, 0.1)
        model_name = MODEL_NAMES[model_type]
        if model_type == "clip":
            model = CLIPLoRAClassifier(model_name, num_classes, lora_cfg).to(device)
        else:
            model = ViTLoRAClassifier(model_name, num_classes, lora_cfg).to(device)

        trainable, total = count_trainable_params(model)
        run_config = {**base_config, "lr": lr}
        metrics = train_model(run_config, model, train_loader, val_loader, device)

        rows.append({
            "model_type": model_type,
            "lora_rank": lora_rank,
            "lr": lr,
            "target_modules": str(target_mods),
            "top1": metrics.get("best_val_top1", 0.0),
            "top5": metrics.get("best_val_top5", 0.0),
            "trainable_params": trainable,
            "total_params": total,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nTop-3 configs by top-1 accuracy:")
    print(df.nlargest(3, "top1")[["model_type", "lora_rank", "lr", "top1", "trainable_params"]])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clip_base.yaml")
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--output_csv", default="results/ablation_results.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_cfg = {
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr": config["training"]["lr"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_ratio": config["training"]["warmup_ratio"],
        "fp16": config["training"].get("fp16", False),
    }
    run_ablation(base_cfg, args.data_root, args.output_csv, device)
