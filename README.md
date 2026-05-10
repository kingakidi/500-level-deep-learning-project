# COEN543 Deep Learning Project — Topic 4

Topic 4: **ShuffleNetV2 pretrained model** finetuned on **Dataset 3** (10 classes), with:
- **Model 1**: ShuffleNetV2 **x1.0** (transfer learning, partial freeze)
- **Model 2**: ShuffleNetV2 **x0.5 + ECA** (lighter model with Efficient Channel Attention)

This project trains both models, evaluates **accuracy / precision / recall / f1-score**, computes **FLOPs**, and saves **Accuracy vs Epochs** plots.

## Folder layout (expected)

Your dataset must be in this structure:

```
deep-learning-project/
  Dataset3/
    train/
      class_1/
        *.jpg
      class_2/
        *.jpg
      ...
    val/
      class_1/
        *.jpg
      class_2/
        *.jpg
      ...
```

## Installation

From `E:\abu-projects\deep-learning-project`:

```bash
pip install -r requirements.txt
```

If you have both `python` and `pip` installed, that’s all you need.

## Run training (Model 1 + Model 2)

```bash
python train_topic4.py --data-dir Dataset3 --models both --epochs 40 --batch-size 64
```

If your machine is slow (CPU-only), start with fewer epochs:

```bash
python train_topic4.py --data-dir Dataset3 --models both --epochs 1 --batch-size 64
```

## Run only one model

Model 1 only:

```bash
python train_topic4.py --data-dir Dataset3 --models 1 --epochs 40
```

Model 2 only:

```bash
python train_topic4.py --data-dir Dataset3 --models 2 --epochs 40
```

## Output files

By default, outputs are saved to `outputs_topic4/`:
- `model1_best.pt`, `model2_best.pt` (best checkpoints on validation accuracy)
- `model1_metrics.json`, `model2_metrics.json` (metrics + training history)
- `model1_accuracy_vs_epochs.png`, `model2_accuracy_vs_epochs.png`
- `improvement_model2_vs_model1.json` (only when training `--models both`)

You can change the output directory:

```bash
python train_topic4.py --data-dir Dataset3 --out-dir outputs_topic4_run2
```

## Useful options

- `--lr 3e-4` (learning rate)
- `--image-size 224` (input image size)
- `--num-workers 2` (data loading workers; set `0` if you get dataloader issues)
- `--no-pretrained` (not recommended; disables ImageNet weights)

