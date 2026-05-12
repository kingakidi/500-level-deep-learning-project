# Deep Learning Project — ShuffleNetV2 on Dataset 3

This project fine-tunes **ShuffleNetV2** on **Dataset 3** (10-class tomato disease images). It trains two models: a **ShuffleNetV2 x1.0** baseline with ImageNet weights and partial freezing, and a lighter **ShuffleNetV2 x0.5** backbone with **ECA** (Efficient Channel Attention). Training saves validation metrics, checkpoints, and accuracy curves.

## Setup

1. Clone or copy this repository and open a terminal at the project root (the folder that contains `main.py` and `requirements.txt`).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the image dataset under `Dataset3/` using the layout in **Folder layout** below.

4. Start training (pick one):

Both models in one run:

```bash
python main.py --data-dir Dataset3 --models both --epochs 80 --batch-size 64
```

Baseline only (ShuffleNetV2 x1.0):

```bash
python main.py --data-dir Dataset3 --models 1 --epochs 80 --batch-size 64
```

ECA variant only (ShuffleNetV2 x0.5 + ECA):

```bash
python main.py --data-dir Dataset3 --models 2 --epochs 80 --batch-size 64
```

## Folder layout

**Dataset (required)**

```
Dataset3/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```

Train and validation splits must use the same class folder names.

**Code**

- `main.py` — training entry point.
- `src/` — model definitions (`models.py`, `eca.py`) and training pipeline (`experiment.py`).

**Outputs**

- `artifacts/` — default location for new runs (`--out-dir` defaults here). Checkpoints, JSON metrics, plots, and `comparison_eca_vs_baseline.json` when both models are trained.
- `runs/dataset3_full_run/` — saved full benchmark runs (e.g. `model1_*`, `model2_*`, `improvement_model2_vs_model1.json`).
- `runs/dataset3_quick_run/` — saved shorter test runs.
