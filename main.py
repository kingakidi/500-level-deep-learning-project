"""Fine-tune ShuffleNetV2 models on an ImageFolder dataset (train/ and val/ subfolders)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment import ExperimentConfig, ShuffleNetComparisonBenchmark


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Train ShuffleNetV2 x1.0 (baseline) and/or x0.5 with ECA on a classification dataset."
        )
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=root / "Dataset3",
        help="Root folder containing train/ and val/ (ImageFolder layout).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Maximum epochs (stops earlier when validation accuracy reaches the target).",
    )
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument(
        "--target-val-acc",
        type=float,
        default=0.98,
        help="Stop training once validation accuracy reaches this threshold.",
    )
    p.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=10,
        help="1-based epoch to unfreeze all layers and switch to phase-2 hyperparameters. Use 0 to disable.",
    )
    p.add_argument("--phase2-lr", type=float, default=1e-4)
    p.add_argument("--phase2-label-smoothing", type=float, default=0.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        type=str,
        default="both",
        choices=("1", "2", "both"),
        help="Train baseline (1), ECA variant (2), or both in one run.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "artifacts",
        help="Directory for checkpoints, plots, and metrics JSON.",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize from random weights instead of ImageNet (not recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_argparse(args)
    ShuffleNetComparisonBenchmark(cfg).run()


if __name__ == "__main__":
    main()
