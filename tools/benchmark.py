import argparse

import plugins

from core.config import load_config, parse_overrides
from core.runtime import load_method_checkpoint, method_requires_checkpoint, resolve_device
from core.registry import DATASET_REGISTRY, METHOD_REGISTRY, METRIC_REGISTRY, lookup
from engine.benchmark_runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run the LLIE benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--opts", nargs="+", default=[], help="Override config values like method=clahe")
    args = parser.parse_args()

    try:
        overrides = parse_overrides(args.opts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_config(args.config, overrides=overrides)

    method = lookup(METHOD_REGISTRY, config["method"])()
    if method_requires_checkpoint(method):
        try:
            _, status = load_method_checkpoint(
                method,
                config["method"],
                ckpt=config.get("ckpt"),
                ckpt_dir=config.get("ckpt_dir"),
                required=True,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(status)
    else:
        print(f"Method '{config['method']}' does not need a checkpoint.")

    dataset_kwargs = {"split": config.get("test_split", "test")}
    if "subset" in config:
        dataset_kwargs["subset"] = config["subset"]

    dataset = lookup(DATASET_REGISTRY, config["dataset"])(
        config["data_root"],
        **dataset_kwargs,
    )
    metrics = [lookup(METRIC_REGISTRY, metric_name)() for metric_name in config["metrics"]]

    device = resolve_device()
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
