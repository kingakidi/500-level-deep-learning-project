"""
COEN543 Topic 4: ShuffleNetV2 on Dataset 3 (10-class tomato disease).
Train Model 1 (ShuffleNetV2 x1.0, partial freeze) vs Model 2 (ShuffleNetV2 x0.5 + ECA).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from thop import profile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import build_model_1, build_model_2, count_parameters

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=25),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def make_loaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_tf, val_tf = build_transforms(image_size)
    train_set = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_set = datasets.ImageFolder(str(val_dir), transform=val_tf)
    if train_set.classes != val_set.classes:
        raise ValueError("Train and val class folders differ; check Dataset3 layout.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_set.classes


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(y_true, y_pred)
    return float(avg_loss), float(acc), y_true, y_pred


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False, desc="train"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


def measure_flops_gflops(model: nn.Module, device: torch.device, image_size: int) -> float:
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        return float("nan")
    return float(flops) / 1e9


def plot_accuracy_vs_epochs(
    history: dict,
    out_path: Path,
    title: str,
) -> None:
    epochs = np.arange(1, len(history["val_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train accuracy", marker="o", markersize=3)
    plt.plot(epochs, history["val_acc"], label="Val accuracy", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    out_dir: Path,
    image_size: int,
) -> dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {"train_acc": [], "val_acc": [], "val_loss": []}
    best_state = None
    best_val = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler
        )
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        dt = time.perf_counter() - t0
        print(
            f"[{name}] epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / f"{name}_best.pt"
    torch.save({"model": best_state, "epoch": best_epoch, "val_acc": best_val}, ckpt_path)

    _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    metrics = classification_metrics(y_true, y_pred)
    metrics["best_val_acc_epoch"] = best_epoch
    metrics["params_total"] = count_parameters(model, trainable_only=False)
    metrics["params_trainable"] = count_parameters(model, trainable_only=True)
    metrics["gflops"] = measure_flops_gflops(model, device, image_size)

    plot_accuracy_vs_epochs(
        history,
        out_dir / f"{name}_accuracy_vs_epochs.png",
        title=f"{name} — accuracy vs epochs",
    )

    with open(out_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_on_val_split": metrics}, f, indent=2)

    return {"history": history, "metrics": metrics}


def relative_improvement(before: float, after: float) -> float:
    if before == 0:
        return float("inf") if after > 0 else 0.0
    return (after - before) / before * 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COEN543 Topic 4 training")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Dataset3",
        help="Folder containing train/ and val/ (ImageFolder layout).",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        type=str,
        default="both",
        choices=("1", "2", "both"),
        help="Train Model 1 only, Model 2 only, or both sequentially.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_topic4",
        help="Checkpoints, plots, and metrics JSON.",
    )
    p.add_argument("--no-pretrained", action="store_true", help="Random init (not recommended).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = make_loaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    pretrained = not args.no_pretrained
    results: dict[str, dict] = {}

    if args.models in ("1", "both"):
        print("\n=== Model 1: ShuffleNetV2 x1.0 (partial freeze) ===")
        m1 = build_model_1(num_classes=num_classes, pretrained=pretrained)
        results["model1"] = train_model(
            "model1",
            m1,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.label_smoothing,
            args.out_dir,
            args.image_size,
        )

    if args.models in ("2", "both"):
        print("\n=== Model 2: ShuffleNetV2 x0.5 + ECA (lighter) ===")
        m2 = build_model_2(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=args.image_size,
        )
        results["model2"] = train_model(
            "model2",
            m2,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.label_smoothing,
            args.out_dir,
            args.image_size,
        )

    if "model1" in results and "model2" in results:
        acc1 = results["model1"]["metrics"]["accuracy"]
        acc2 = results["model2"]["metrics"]["accuracy"]
        summary = {
            "model1_accuracy": acc1,
            "model2_accuracy": acc2,
            "relative_improvement_accuracy_pct": relative_improvement(acc1, acc2),
            "model1_f1_macro": results["model1"]["metrics"]["f1_macro"],
            "model2_f1_macro": results["model2"]["metrics"]["f1_macro"],
            "relative_improvement_f1_macro_pct": relative_improvement(
                results["model1"]["metrics"]["f1_macro"],
                results["model2"]["metrics"]["f1_macro"],
            ),
            "model1_params_total": results["model1"]["metrics"]["params_total"],
            "model2_params_total": results["model2"]["metrics"]["params_total"],
            "model1_gflops": results["model1"]["metrics"]["gflops"],
            "model2_gflops": results["model2"]["metrics"]["gflops"],
        }
        print("\n=== Improvement (Model 2 vs Model 1) ===")
        print(json.dumps(summary, indent=2))
        with open(args.out_dir / "improvement_model2_vs_model1.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"\nArtifacts saved under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
