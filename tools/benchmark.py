import argparse

import torch

from core.checkpoint import resolve_checkpoint_path
from core.config import load_config, parse_overrides
from core.registry import DATASET_REGISTRY, METHOD_REGISTRY, METRIC_REGISTRY, lookup
from engine.benchmark_runner import BenchmarkRunner

# import plugins so they register themselves
import plugins


def main():
    parser = argparse.ArgumentParser(description="Run LLIE benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--opts", nargs="+", default=[], help="Override config values e.g. method=clahe lr=0.0002")
    args = parser.parse_args()

    overrides = parse_overrides(args.opts)
    config = load_config(args.config, overrides=overrides)

    method = lookup(METHOD_REGISTRY, config["method"])()

    ckpt_path = resolve_checkpoint_path(config.get("ckpt"), config.get("ckpt_dir"))
    if ckpt_path:
        print(f"Auto-loading checkpoint: {ckpt_path}")
        method.load_ckpt(str(ckpt_path))
    else:
        print("No checkpoint; using random weights")

    dataset_kwargs = {
        "split": config.get("test_split", "test"),
    }
    if "subset" in config:
        dataset_kwargs["subset"] = config["subset"]

    dataset = lookup(DATASET_REGISTRY, config["dataset"])(
        config["data_root"],
        **dataset_kwargs,
    )
    metrics = [lookup(METRIC_REGISTRY, metric_name)() for metric_name in config["metrics"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    runner = BenchmarkRunner(
        methods=[method],
        datasets=[dataset],
        metrics=metrics,
        device=device,
        log_dir=config["log_dir"],
        protocol_path="docs/benchmark_protocol.md",
        batch_size=config.get("eval_batch_size", config.get("batch_size", 1)),
    )
    runner.run()


if __name__ == "__main__":
    main()
