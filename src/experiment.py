"""ShuffleNetV2 fine-tuning benchmark: baseline x1.0 vs x0.5 + ECA on ImageFolder data."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
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

from src.models import (
    build_shufflenet_baseline,
    build_shufflenet_eca_variant,
    count_parameters,
    unfreeze_all,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    out_dir: Path
    models: Literal["1", "2", "both"]
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    target_val_acc: float
    unfreeze_epoch: int
    phase2_lr: float
    phase2_label_smoothing: float
    image_size: int
    num_workers: int
    seed: int
    use_pretrained: bool

    @staticmethod
    def from_argparse(ns: object) -> ExperimentConfig:
        return ExperimentConfig(
            data_dir=ns.data_dir,
            out_dir=ns.out_dir,
            models=ns.models,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            lr=ns.lr,
            weight_decay=ns.weight_decay,
            label_smoothing=ns.label_smoothing,
            target_val_acc=ns.target_val_acc,
            unfreeze_epoch=ns.unfreeze_epoch,
            phase2_lr=ns.phase2_lr,
            phase2_label_smoothing=ns.phase2_label_smoothing,
            image_size=ns.image_size,
            num_workers=ns.num_workers,
            seed=ns.seed,
            use_pretrained=not ns.no_pretrained,
        )


class ImageFolderFineTuneDataModule:
    def __init__(
        self,
        data_root: Path,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _train_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomRotation(degrees=25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _val_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def dataloaders(self) -> tuple[DataLoader, DataLoader, list[str]]:
        train_dir = self.data_root / "train"
        val_dir = self.data_root / "val"
        train_set = datasets.ImageFolder(str(train_dir), transform=self._train_transforms())
        val_set = datasets.ImageFolder(str(val_dir), transform=self._val_transforms())
        if train_set.classes != val_set.classes:
            raise ValueError("Train and validation class folders must match.")

        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin,
        )
        return train_loader, val_loader, train_set.classes


@torch.no_grad()
def _evaluate_batch_metrics(
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


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _train_one_epoch(
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


def _gflops(model: nn.Module, device: torch.device, image_size: int) -> float:
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        return float("nan")
    return float(flops) / 1e9


def _plot_accuracy_history(
    history: dict[str, list[float]],
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


def _relative_gain(before: float, after: float) -> float:
    if before == 0:
        return float("inf") if after > 0 else 0.0
    return (after - before) / before * 100.0


class ShuffleNetFineTuneSession:
    def __init__(
        self,
        run_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        out_dir: Path,
        image_size: int,
        max_epochs: int,
        lr: float,
        weight_decay: float,
        label_smoothing: float,
        target_val_acc: float,
        unfreeze_epoch: int,
        phase2_lr: float,
        phase2_label_smoothing: float,
    ) -> None:
        self.run_name = run_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = out_dir
        self.image_size = image_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.target_val_acc = target_val_acc
        self.unfreeze_epoch = unfreeze_epoch
        self.phase2_lr = phase2_lr
        self.phase2_label_smoothing = phase2_label_smoothing

    def execute(self, model: nn.Module) -> dict[str, object]:
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1)
        )
        scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        history: dict[str, list[float]] = {"train_acc": [], "val_acc": [], "val_loss": []}
        best_state = None
        best_val = -1.0
        best_epoch = -1
        stopped_reason = "max_epochs"
        tag = self.run_name

        for epoch in range(1, self.max_epochs + 1):
            if self.unfreeze_epoch > 0 and epoch == self.unfreeze_epoch:
                unfreeze_all(model)
                criterion = nn.CrossEntropyLoss(label_smoothing=self.phase2_label_smoothing)
                remaining = self.max_epochs - epoch + 1
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.phase2_lr,
                    weight_decay=self.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(remaining, 1)
                )
                print(
                    f"[{tag}] epoch {epoch}: full unfreeze, "
                    f"lr={self.phase2_lr}, label_smoothing={self.phase2_label_smoothing}"
                )

            t0 = time.perf_counter()
            train_loss, train_acc = _train_one_epoch(
                model, self.train_loader, self.device, criterion, optimizer, scaler
            )
            val_loss, val_acc, y_true, y_pred = _evaluate_batch_metrics(
                model, self.val_loader, self.device, criterion
            )
            scheduler.step()
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            dt = time.perf_counter() - t0
            print(
                f"[{tag}] epoch {epoch}/{self.max_epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)"
            )
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if val_acc >= self.target_val_acc:
                stopped_reason = f"target_val_acc>={self.target_val_acc}"
                print(f"[{tag}] early stop: validation accuracy reached {val_acc:.4f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        ckpt_path = self.out_dir / f"{tag}_best.pt"
        torch.save(
            {
                "model": best_state,
                "epoch": best_epoch,
                "val_acc": best_val,
                "stopped_reason": stopped_reason,
                "target_val_acc": self.target_val_acc,
            },
            ckpt_path,
        )

        _, _, y_true, y_pred = _evaluate_batch_metrics(
            model, self.val_loader, self.device, criterion
        )
        metrics = _classification_report(y_true, y_pred)
        metrics["best_val_acc_epoch"] = best_epoch
        metrics["best_val_acc"] = float(best_val)
        metrics["target_val_acc"] = float(self.target_val_acc)
        metrics["met_target"] = bool(metrics["accuracy"] >= self.target_val_acc)
        metrics["stopped_reason"] = stopped_reason
        metrics["params_total"] = count_parameters(model, trainable_only=False)
        metrics["params_trainable"] = count_parameters(model, trainable_only=True)
        metrics["gflops"] = _gflops(model, self.device, self.image_size)

        _plot_accuracy_history(
            history,
            self.out_dir / f"{tag}_accuracy_vs_epochs.png",
            title=f"{tag} — accuracy vs epochs",
        )

        payload = {"history": history, "validation_split_metrics": metrics}
        with open(self.out_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if not metrics["met_target"]:
            print(
                f"[{tag}] WARNING: final val accuracy {metrics['accuracy']:.4f} "
                f"< target {self.target_val_acc}. "
                "Consider more epochs, GPU, or tuning --lr / --phase2-lr."
            )

        return {"history": history, "metrics": metrics}


class ShuffleNetComparisonBenchmark:
    BASELINE_TAG = "shufflenet_x10_baseline"
    ECA_TAG = "shufflenet_x05_eca"

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        set_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        data = ImageFolderFineTuneDataModule(
            self.cfg.data_dir,
            self.cfg.image_size,
            self.cfg.batch_size,
            self.cfg.num_workers,
        )
        train_loader, val_loader, class_names = data.dataloaders()
        num_classes = len(class_names)
        print(f"Classes ({num_classes}): {class_names}")

        pretrained = self.cfg.use_pretrained
        results: dict[str, dict[str, object]] = {}

        hparams = dict(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=self.cfg.out_dir,
            image_size=self.cfg.image_size,
            max_epochs=self.cfg.epochs,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            label_smoothing=self.cfg.label_smoothing,
            target_val_acc=self.cfg.target_val_acc,
            unfreeze_epoch=self.cfg.unfreeze_epoch,
            phase2_lr=self.cfg.phase2_lr,
            phase2_label_smoothing=self.cfg.phase2_label_smoothing,
        )

        if self.cfg.models in ("1", "both"):
            print("\n=== ShuffleNetV2 x1.0 (ImageNet, partial freeze) ===")
            m1 = build_shufflenet_baseline(num_classes=num_classes, pretrained=pretrained)
            session = ShuffleNetFineTuneSession(run_name=self.BASELINE_TAG, **hparams)
            results["shufflenet_x10_baseline"] = session.execute(m1)

        if self.cfg.models in ("2", "both"):
            print("\n=== ShuffleNetV2 x0.5 + ECA ===")
            m2 = build_shufflenet_eca_variant(
                num_classes=num_classes,
                pretrained=pretrained,
                image_size=self.cfg.image_size,
            )
            session = ShuffleNetFineTuneSession(run_name=self.ECA_TAG, **hparams)
            results["shufflenet_x05_eca"] = session.execute(m2)

        bkey, ekey = "shufflenet_x10_baseline", "shufflenet_x05_eca"
        if bkey in results and ekey in results:
            acc_b = results[bkey]["metrics"]["accuracy"]  # type: ignore[index]
            acc_e = results[ekey]["metrics"]["accuracy"]  # type: ignore[index]
            f1_b = results[bkey]["metrics"]["f1_macro"]  # type: ignore[index]
            f1_e = results[ekey]["metrics"]["f1_macro"]  # type: ignore[index]
            summary = {
                "baseline_shufflenet_x10_val_accuracy": acc_b,
                "eca_shufflenet_x05_val_accuracy": acc_e,
                "relative_improvement_val_accuracy_pct": _relative_gain(acc_b, acc_e),
                "baseline_f1_macro": f1_b,
                "eca_variant_f1_macro": f1_e,
                "relative_improvement_f1_macro_pct": _relative_gain(f1_b, f1_e),
                "baseline_params_total": results[bkey]["metrics"]["params_total"],  # type: ignore[index]
                "eca_params_total": results[ekey]["metrics"]["params_total"],  # type: ignore[index]
                "baseline_gflops": results[bkey]["metrics"]["gflops"],  # type: ignore[index]
                "eca_gflops": results[ekey]["metrics"]["gflops"],  # type: ignore[index]
            }
            print("\n=== ECA variant vs baseline (validation split) ===")
            print(json.dumps(summary, indent=2))
            out_json = self.cfg.out_dir / "comparison_eca_vs_baseline.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        print(f"\nArtifacts saved under: {self.cfg.out_dir.resolve()}")
