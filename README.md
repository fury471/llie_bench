# LLIE Bench

A modular benchmark framework for Low-Light Image Enhancement.

## Table of Contents

- [Quick Start](#quick-start)
- [Pipeline Internals](#pipeline-internals)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Setup](#setup)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Adding New Plugins](#adding-new-plugins)

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data
See [Data Setup](#data-setup) section.

### 3. Train a method
```bash
python -m tools.train --config configs/experiments/full_bench_lolv1.yaml
```

### 4. Run inference on a single image
```bash
python -m tools.inference --method zerodce --ckpt checkpoints/zerodce_lolv1/checkpoint_epoch_099.pth --input your_image.png --output results/enhanced.png
```

### 5. Run benchmark
```bash
python -m tools.benchmark --config configs/experiments/full_bench_lolv1.yaml
```

### 6. Run with traditional method (no training needed)
```bash
python -m tools.inference --method clahe --input your_image.png --output results/enhanced.png
```

### 7. Run the interactive demo
```bash
python -m tools.demo
```

### 8. Export results table
```bash
python -m tools.export_tables --log_dir logs/zerodce_lolv1/eval --output results/table.csv
```

## Pipeline Internals

### Training Pipeline

```mermaid
flowchart TD
    Start([Execute:
tools/train.py]) --> Cfg
    
    subgraph Setup Phase
        Cfg[load_config: 
        Merge YAMLs] --> RM[lookup METHOD_REGISTRY]
        RM --> RD[lookup DATASET_REGISTRY]
        RD --> DL[DataLoader: 
        Batch Training Data]
        DL --> Tr[Initialize Trainer]
        Tr --> Seed[set_seed: 
        Reproducibility]
    end

    Seed --> EpochLoop{For Each Epoch}

    subgraph Training Loop
        Loss[model.compute_loss] --> ZG[optimizer.zero_grad]
        ZG --> Bwd[loss.backward]
        Bwd --> Step[optimizer.step]
    end

    subgraph Logging & Saving
        Log[Logger.log: 
        Save to CSV] --> Ckpt[CheckpointManager.save]
    end

    EpochLoop --> Loss
    Step --> Log
    Ckpt --> |Next Epoch| EpochLoop
    Ckpt --> |Max Epochs Reached| Finish([End Training])
```

### Benchmark Pipeline

```mermaid
flowchart TD
    Start([Execute:
tools/benchmark.py]) --> Cfg
    
    subgraph Setup Phase
        Cfg[load_config: 
        Merge YAMLs] --> Ckpt[Auto-load latest checkpoint]
        Ckpt --> BR[Initialize BenchmarkRunner]
    end

    BR --> MethodLoop{For Each Method}
    MethodLoop --> |Next| DatasetLoop{For Each Dataset}

    subgraph Evaluation Pipeline
        Compat[Protocol.check_compatibility: 
        Enforce fairness] --> Eval[Initialize Evaluator]
        Eval --> Infer[model.enhance: 
        Batch inference]
        Infer --> MetricLoop{For Each Metric}
        
        MetricLoop --> Compute[metric.compute: 
        Per-image score]
        Compute --> |Next Metric| MetricLoop
        MetricLoop --> |All metrics done| Agg[metric.aggregate: 
        Dataset mean]
    end

    DatasetLoop --> Compat
    Agg --> |Next Dataset| DatasetLoop
    DatasetLoop --> |All datasets done| MethodLoop
    
    MethodLoop --> |All methods done| Log[Logger.log: Save to CSV]
    Log --> Finish([End Benchmark])
```

### ### Inference Pipeline

```mermaid
flowchart TD
    Start([Execute:
tools/inference.py]) --> Lookup

    subgraph Setup Phase
        Lookup[lookup METHOD_REGISTRY: 
        Find method] --> Ckpt[method.load_ckpt: 
        Load weights]
    end

    Ckpt --> LoadImg

    subgraph Image Processing
        LoadImg[cv2.imread: 
        Load input image] --> Norm[Normalize to 0-1 
        & Convert to Tensor]
        Norm --> Enhance[method.enhance: 
        Run enhancement]
        Enhance --> Save[cv2.imwrite: 
        Save output image]
    end

    Save --> Finish([End Inference])
```

### Export Pipeline

```mermaid
flowchart TD
    Start([Execute: 
    tools/export_tables.py]) --> Read

    subgraph Data Processing
        Read[pd.read_csv: 
        Read logs CSV] --> Pivot[pivot_table: 
        Reshape to Method x Metric]
    end

    Pivot --> Output[Print & Save to CSV]
    
    Output --> Finish([End Export])
```

### Demo Pipeline

```mermaid
flowchart TD
    Start([Execute: 
    tools/export_tables.py]) --> Read

    subgraph Data Processing
        Read[pd.read_csv: 
        Read logs CSV] --> Pivot[pivot_table: 
        Reshape to Method x Metric]
    end

    Pivot --> Output[Print & Save to CSV]
    
    Output --> Finish([End Export])
```

## Project Structure

- `tools/` — entry points (CLI scripts)
- `engine/` — trainer, evaluator, benchmark runner
- `core/` — registry, config, logger, checkpoint, seed, protocol
- `methods/` — enhancement algorithm plugins
- `datasets_loaders/` — dataset plugins
- `metrics/` — metric plugins
- `configs/` — YAML configs for experiments, methods, datasets, metrics
- `docs/` — documentation

## Architecture

The project is built on 5 layers:

| Layer                    | Folder(s)                                 | Responsibility                                       |
| ------------------------ | ----------------------------------------- | ---------------------------------------------------- |
| Layer 1 — Entry Points   | `tools/`                                  | CLI scripts, thin wrappers                           |
| Layer 2 — Engine         | `engine/`                                 | Training and evaluation logic                        |
| Layer 3 — Core Framework | `core/`                                   | Registry, config, logger, checkpoint, seed, protocol |
| Layer 4 — Plugins        | `methods/` `datasets_loaders/` `metrics/` | Algorithms, datasets, metrics                        |
| Layer 5 — Infrastructure | `configs/` `docs/`                        | YAML configs, documentation                          |

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

### LOL-v2
1. Download from: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
2. Extract and place under `data/LOL-v2/`
3. Expected structure:
```
data/LOL-v2/LOL-v2/
├── Real_captured/
│   ├── Train/
│   │   ├── Low/
│   │   └── Normal/
│   └── Test/
│       ├── Low/
│       └── Normal/
└── Synthetic/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

## Usage

```bash
python -m tools.benchmark --config configs/experiments/full_bench_lolv1.yaml
```

## Adding New Plugins

### Rules

- Every method must inherit `BaseMethod`

- Every dataset must inherit `BaseDataset`

- Every metric must inherit `BaseMetric`

- Always register using `@REGISTRY.register("name")` decorator

- Always add a YAML config in the corresponding `configs/` subfolder


### Adding a New Method

**Step 1** — create `methods/your_method.py`:

```python
from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

@METHOD_REGISTRY.register("your_method")
class YourMethod(BaseMethod):
    def __init__(self):
        super().__init__()
        # define your model here

    def forward(self, batch):
        # run model on batch[0], return raw output
        pass

    def enhance(self, batch):
        # return final enhanced image
        # for simple methods: return self.forward(batch)
        pass

    def compute_loss(self, batch):
        # return training loss
        # for traditional methods: raise NotImplementedError
        pass

    def load_ckpt(self, path):
        # load weights from path
        # for traditional methods: pass
        pass

    def get_meta(self):
        return {
            "name": "your_method",
            "type": "srgb",  # or "raw"
            "paired": True,  # or False
        }
```

**Step 2** — create `configs/methods/your_method.yaml`:
```yaml
method: your_method
lr: 0.0001
epochs: 100
batch_size: 8
```

### Adding a New Dataset

**Step 1** — create `datasets_loaders/your_dataset.py`:

```python
from datasets_loaders.base import BaseDataset
from core.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("your_dataset")
class YourDataset(BaseDataset):
    def __init__(self, root, split="train"):
        super().__init__()
        # load file paths here

    def __len__(self):
        # return number of samples
        pass

    def __getitem__(self, idx):
        # return [low, high] tensors
        # shape: (3, H, W), normalized to [0, 1]
        pass

    def get_meta(self):
        return {
            "name": "your_dataset",
            "type": "srgb",       # or "raw"
            "paired": True,       # or False
            "eval_channel": "y",  # or "rgb"
        }
```

**Step 2** — create `configs/datasets/your_dataset.yaml`:
```yaml
dataset: your_dataset
data_root: data/your_dataset
test_split: test
```

### Adding a New Metric

**Step 1** — create `metrics/your_metric.py`:

```python
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register("your_metric")
class YourMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self, pred, gt):
        # compute score for one image pair
        # store result in self.values
        pass

    def aggregate(self):
        # return mean score across all images
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []
```

**Step 2** — create `configs/metrics/your_metric.yaml`:

```yaml
metrics:
  - your_metric
```

### Key Rule
> The engine never changes. Only plugins change.
