# LLIE Bench

A modular benchmark framework for Low-Light Image Enhancement.

## Project Structure

- `tools/` — entry points (CLI scripts)
- `engine/` — trainer, evaluator, benchmark runner
- `core/` — registry, config, logger, checkpoint, seed, protocol
- `methods/` — enhancement algorithm plugins
- `datasets/` — dataset plugins
- `metrics/` — metric plugins
- `configs/` — YAML configs for experiments, methods, datasets, metrics
- `docs/` — documentation

## Architecture

The project is built on 5 layers:

| Layer                    | Folder(s)                         | Responsibility                                       |
| ------------------------ | --------------------------------- | ---------------------------------------------------- |
| Layer 1 — Entry Points   | `tools/`                          | CLI scripts, thin wrappers                           |
| Layer 2 — Engine         | `engine/`                         | Training and evaluation logic                        |
| Layer 3 — Core Framework | `core/`                           | Registry, config, logger, checkpoint, seed, protocol |
| Layer 4 — Plugins        | `methods/` `datasets/` `metrics/` | Algorithms, datasets, metrics                        |
| Layer 5 — Infrastructure | `configs/` `docs/`                | YAML configs, documentation                          |

## Design Principle

> Write each piece of logic once, in one place, and never touch it again when things change.

This is called **low coupling**. The engine never knows what method, dataset, or metric it is talking to — it only knows abstract interfaces.

## Setup

```bash
pip install -r requirements.txt
```

## Data Setup

### LOLv1
1. Download from: https://daooshee.github.io/BMVC2018website/
2. Extract and place under `data/LOLdataset/`
3. Expected structure:
```
data/LOLdataset/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

## Usage

```bash
python tools/benchmark.py --config configs/experiments/full_bench_lolv1.yaml
```

## Adding New Plugins

This framework is designed to be extended with minimal effort.

### Adding a New Method
1. Create `methods/your_method.py` inheriting `BaseMethod`
2. Implement `forward`, `enhance`, `compute_loss`, `load_ckpt`, `get_meta`
3. Register: `METHOD_REGISTRY["your_method"] = YourMethod`
4. Add `configs/methods/your_method.yaml`

### Adding a New Dataset
1. Create `datasets/your_dataset.py` inheriting `BaseDataset`
2. Implement `__getitem__`, `__len__`, `get_meta`
3. Register: `DATASET_REGISTRY["your_dataset"] = YourDataset`
4. Add `configs/datasets/your_dataset.yaml`

### Adding a New Metric
1. Create `metrics/your_metric.py` inheriting `BaseMetric`
2. Implement `compute`, `aggregate`, `reset`
3. Register: `METRIC_REGISTRY["your_metric"] = YourMetric`
4. Add `configs/metrics/your_metric.yaml`

### Key Rule
> The engine never changes. Only plugins change.