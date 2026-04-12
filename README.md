# LLIE Bench

A modular benchmark framework for low-light image enhancement.

The project is built around a plugin architecture:

- `methods/` contains enhancement algorithms
- `datasets_loaders/` contains dataset adapters
- `metrics/` contains evaluation metrics
- `engine/` contains generic training and evaluation loops
- `core/` contains shared framework utilities such as config loading, registries, logging, checkpoints, and protocol checks

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Config Precedence](#config-precedence)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Pipeline Internals](#pipeline-internals)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Adding New Plugins](#adding-new-plugins)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

See [Data Setup](#data-setup).

### 3. Train a method

Use an experiment config for reproducibility:

```bash
python -m tools.train --config configs/experiments/full_bench_lolv1.yaml
```

Override a few values on the fly:

```bash
python -m tools.train --config configs/experiments/full_bench_lolv1.yaml --opts lr=0.0002 batch_size=4
```

If you switch to a different trainable method, prefer a dedicated experiment config or override all method-specific knobs you care about, such as `batch_size`, `patch_size`, `random_flip`, and `train_phase`.

### 4. Run inference on a single image

```bash
python -m tools.inference --method zerodce --ckpt checkpoints/zerodce_lolv1/checkpoint_epoch_099.pth --input your_image.png --output results/enhanced.png
```

You can also pass `--ckpt_dir` instead of `--ckpt`. The tool will auto-load the latest checkpoint in that directory.

Learned methods require a checkpoint for inference. Traditional methods such as `clahe` do not.

### 5. Run benchmark

```bash
python -m tools.benchmark --config configs/experiments/full_bench_lolv1.yaml
```

Benchmark evaluation uses `eval_batch_size` when present in config, otherwise it falls back to `batch_size`.

Learned methods require `ckpt` or `ckpt_dir` for benchmarking. Traditional methods do not.

### 6. Run with a traditional method

```bash
python -m tools.inference --method clahe --input your_image.png --output results/enhanced.png
```

### 7. Export results table

```bash
python -m tools.export_tables --log_dir logs/zerodce_lolv1/eval --output results/table.csv
```

### 8. Run the web app

```bash
python -m tools.app
```

The web app method list is built from the registry, so newly registered methods appear automatically.

## CLI Reference

Most scripts in this project follow the same pattern:

```bash
python -m tools.<script_name> [arguments]
```

The web app entrypoint follows the same style:

```bash
python -m tools.app [arguments]
```

### `--config`

Used by `tools.train` and `tools.benchmark`.

It points to an experiment YAML file:

```bash
python -m tools.train --config configs/experiments/full_bench_lolv1.yaml
```

The experiment config is the main entry point for reproducible runs.

### `--opts`

Used by `tools.train` and `tools.benchmark`.

It applies one or more last-mile overrides on top of the loaded config. Each override must be written as `key=value`.

Example:

```bash
python -m tools.train --config configs/experiments/full_bench_lolv1.yaml --opts lr=0.0002 batch_size=4
```

Multiple overrides are written as space-separated `key=value` pairs after a single `--opts`.

Common examples:

```bash
--opts method=retinexnet
--opts batch_size=2 patch_size=96
--opts train_phase=joint ckpt=checkpoints/retinexnet_lolv1_decom/checkpoint_epoch_020.pth
```

### `--method`

Used by `tools.inference`.

It selects the registered method plugin:

```bash
python -m tools.inference --method zerodce --ckpt checkpoints/zerodce_lolv1/checkpoint_epoch_099.pth --input your_image.png --output results/enhanced.png
```

### `--ckpt`

Used by `tools.inference`, and also accepted through config for training and benchmarking.

It points to a specific checkpoint file:

```bash
--ckpt checkpoints/zerodce_lolv1/checkpoint_epoch_099.pth
```

Use this when you want an exact checkpoint, not just the latest one.

For learned methods, `tools.inference` and `tools.benchmark` expect a checkpoint to be available.

### `--ckpt_dir`

Used by `tools.inference`, and also accepted through config for benchmarking.

It points to a checkpoint directory. The tool will auto-load the latest `.pth` file in that directory:

```bash
--ckpt_dir checkpoints/zerodce_lolv1
```

### `--input` and `--output`

Used by `tools.inference`.

- `--input` is the input image path
- `--output` is where the enhanced image will be saved

Example:

```bash
python -m tools.inference --method clahe --input data/LOLdataset/eval15/low/1.png --output results/clahe_1.png
```

### `--share`, `--host`, and `--port`

Used by `tools.app`.

- `--share` asks Gradio to create a public share link
- `--host` controls which interface address the app binds to
- `--port` controls which TCP port the app uses

Examples:

```bash
python -m tools.app
python -m tools.app --share
python -m tools.app --host 0.0.0.0 --port 8080
```

## Config Precedence

Configuration is resolved in this order:

1. Method, dataset, and metric YAMLs provide defaults.
2. The experiment YAML overrides those defaults.
3. CLI `--opts` overrides everything else.

This means experiment configs are authoritative, and `--opts` should be treated as explicit last-mile overrides.

## Project Structure

- `tools/`: entry points
- `engine/`: trainer, evaluator, benchmark runner
- `core/`: registry, config, logger, checkpoint, protocol, seed, transforms
- `methods/`: enhancement algorithm plugins
- `datasets_loaders/`: dataset plugins
- `metrics/`: metric plugins
- `configs/`: YAML configs for experiments, methods, datasets, and metrics
- `docs/`: documentation and benchmark protocol

## Architecture

The project is organized into five layers:

| Layer | Folder(s) | Responsibility |
| --- | --- | --- |
| Entry Points | `tools/` | Thin CLI and web app wrappers |
| Engine | `engine/` | Generic training and evaluation logic |
| Core Framework | `core/` | Shared utilities and framework contracts |
| Plugins | `methods/`, `datasets_loaders/`, `metrics/` | Algorithms, datasets, metrics |
| Infrastructure | `configs/`, `docs/` | Configuration and documentation |

## Design Principle

> Write each piece of logic once, in one place, and avoid changing the engine for method-specific behavior.

The engine should only contain capabilities that are generic across methods, datasets, and metrics. Algorithm-specific behavior belongs in plugins and configuration.

## Pipeline Internals

### Training Pipeline

```text
tools/train.py
  |
  +-- load_config(...)
  |     |
  |     +-- method/dataset/metric defaults
  |     +-- experiment config
  |     +-- CLI --opts overrides
  |
  +-- build_transforms(config)
  |
  +-- lookup METHOD_REGISTRY
  +-- lookup DATASET_REGISTRY
  +-- DataLoader
  |
  +-- optional method.set_phase(...)
  +-- optional method.load_ckpt(...)
  |
  +-- Trainer.train(...)
        |
        +-- move batch to device
        +-- model.compute_loss(batch)
        +-- backward + optimizer.step
        +-- log loss
        +-- save checkpoint
```

### Benchmark Pipeline

```text
tools/benchmark.py
  |
  +-- load_config(...)
  +-- lookup METHOD_REGISTRY
  +-- optional/required checkpoint load
  +-- lookup DATASET_REGISTRY
  +-- lookup METRIC_REGISTRY
  |
  +-- BenchmarkRunner.run(...)
        |
        +-- Protocol.check_compatibility(...)
        +-- DataLoader
        +-- Evaluator.evaluate(...)
              |
              +-- move batch to device
              +-- model.enhance(batch)
              +-- metric.compute(...)
              +-- metric.aggregate(...)
              +-- log evaluation results
```

### Inference Pipeline

```text
tools/inference.py
  |
  +-- lookup METHOD_REGISTRY
  +-- optional/required checkpoint load
  +-- read input image
  +-- convert RGB image -> BCHW tensor
  +-- method.enhance([img_tensor])
  +-- convert tensor -> RGB image
  +-- save output image
```

### Web App Pipeline

```text
tools/app.py
  |
  +-- build Gradio interface
  +-- lookup METHOD_REGISTRY
  +-- validate checkpoint input
  +-- load checkpoint when needed
  +-- choose device
  +-- try progressively smaller image sizes if memory is tight
  +-- method.enhance([img_tensor])
  +-- convert tensor -> RGB image
  +-- return image + status message to the UI
```

## Data Setup

### LOLv1

1. Download from: https://daooshee.github.io/BMVC2018website/
2. Extract and place under `data/LOLdataset/`
3. Expected structure:

```text
data/LOLdataset/
  our485/
    low/
    high/
  eval15/
    low/
    high/
```

### LOL-v2

1. Download from: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
2. Extract and place under `data/LOL-v2/`
3. Expected structure:

```text
data/LOL-v2/LOL-v2/
  Real_captured/
    Train/
      Low/
      Normal/
    Test/
      Low/
      Normal/
  Synthetic/
    Train/
      Low/
      Normal/
    Test/
      Low/
      Normal/
```

## Usage

### Generic Benchmark Run

```bash
python -m tools.benchmark --config configs/experiments/full_bench_lolv1.yaml
```

### RetinexNet Workflow

RetinexNet supports three phases:

- `decom`: decomposition-led training with a light relight warm-start
- `joint`: full end-to-end training of decomposition and relighting
- `relight`: freeze `DecomNet` and fine-tune only `RelightNet`

Recommended order:

1. Train `decom` first.
2. Resume from a saved checkpoint with `train_phase=joint`.
3. Optionally finish with `train_phase=relight`.

Example:

```bash
# Stage 1: decomposition-led warmup
python -m tools.train --config configs/experiments/retinexnet_lolv1_decom.yaml

# Stage 2: full joint training
python -m tools.train --config configs/experiments/retinexnet_lolv1_decom.yaml --opts train_phase=joint ckpt=checkpoints/retinexnet_lolv1_decom/checkpoint_epoch_020.pth log_dir=logs/retinexnet_lolv1_joint ckpt_dir=checkpoints/retinexnet_lolv1_joint

# Stage 3: optional relight-only fine-tuning
python -m tools.train --config configs/experiments/retinexnet_lolv1_relight.yaml
```

## Adding New Plugins

### Rules

- Every method must inherit `BaseMethod`
- Every dataset must inherit `BaseDataset`
- Every metric must inherit `BaseMetric`
- Always register classes with the appropriate registry decorator
- Always add a YAML config in the corresponding `configs/` subfolder
- Prefer putting algorithm-specific behavior in the plugin itself. Only move logic into `engine/` or `core/` when it is genuinely generic

### Adding a New Method

Step 1: create `methods/your_method.py`

```python
from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("your_method")
class YourMethod(BaseMethod):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass

    def enhance(self, batch):
        pass

    def compute_loss(self, batch):
        pass

    def load_ckpt(self, path):
        pass

    def get_meta(self):
        return {
            "name": "your_method",
            "type": "srgb",
            "paired": True,
        }
```

If your method needs staged training, it may also optionally expose `set_phase(...)`. `tools/train.py` will call it when `train_phase` is present in config.

Step 2: create `configs/methods/your_method.yaml`

```yaml
method: your_method
lr: 0.0001
epochs: 100
batch_size: 8
```

Method configs act as defaults. Experiment configs can override them, and CLI `--opts` can override both.

### Adding a New Dataset

Step 1: create `datasets_loaders/your_dataset.py`

```python
from datasets_loaders.base import BaseDataset
from core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("your_dataset")
class YourDataset(BaseDataset):
    def __init__(self, root, split="train", transforms=None):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_meta(self):
        return {
            "name": "your_dataset",
            "type": "srgb",
            "paired": True,
            "eval_channel": "y",
        }
```

Step 2: create `configs/datasets/your_dataset.yaml`

```yaml
dataset: your_dataset
data_root: data/your_dataset
test_split: test
```

### Adding a New Metric

Step 1: create `metrics/your_metric.py`

```python
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("your_metric")
class YourMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self, pred, gt):
        pass

    def aggregate(self):
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []
```

### Registering Your Plugin

After creating your plugin file, add it to `plugins.py`:

```python
# Methods
import methods.your_method

# Datasets
import datasets_loaders.your_dataset

# Metrics
import metrics.your_metric
```

This ensures the plugin is registered whenever a tool script imports `plugins`.
