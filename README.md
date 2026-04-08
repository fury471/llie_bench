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

## Usage

```bash
python tools/benchmark.py --config configs/experiments/full_bench_lolv1.yaml
```