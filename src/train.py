import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup


def train_one_epoch(model, loader, optimizer, criterion, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    preds = all_logits.argmax(dim=1)
    top1 = (preds == all_labels).float().mean().item()

    k = min(5, all_logits.size(1))
    top5_preds = all_logits.topk(k, dim=1).indices
    top5 = (top5_preds == all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"top1": top1, "top5": top5}


def train_model(config: dict, model, train_loader, val_loader, device) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    total_steps = config["epochs"] * len(train_loader)
    warmup = int(config["warmup_ratio"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = GradScaler(enabled=config.get("fp16", False))

    best_val_top1 = 0.0
    best_metrics = {}
    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        if metrics["top1"] > best_val_top1:
            best_val_top1 = metrics["top1"]
            best_metrics = {"best_val_top1": metrics["top1"], "best_val_top5": metrics["top5"]}

    return best_metrics
