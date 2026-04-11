import torch
import argparse
from core.config import load_config
from core.registry import METHOD_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY, lookup
from engine.benchmark_runner import BenchmarkRunner

# import plugins so they register themselves
import plugins

# all the logic here
def main():
    # parse the --config argument.
    parser = argparse.ArgumentParser(description="Run LLIE benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--opts", nargs="+", default=[], help="Override config values e.g. method=clahe lr=0.0002")
    args = parser.parse_args()

    #  load the config and merge with defaults
    config = load_config(args.config)
    for opt in args.opts:
        key, value = opt.split("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        config[key] = value
        
    # lookup the method, dataset, and metric classes
    method = lookup(METHOD_REGISTRY, config["method"])()
    dataset = lookup(DATASET_REGISTRY, config["dataset"])(config["data_root"], split=config.get("test_split", "test"))
    metrics = [lookup(METRIC_REGISTRY, m)() for m in config["metrics"]]

    # detect the device automatically (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # create the benchmark runner and run the benchmark
    runner = BenchmarkRunner(
        methods = [method],
        datasets = [dataset],
        metrics = metrics,
        device = device,
        log_dir=config["log_dir"],
        protocol_path="docs/benchmark_protocol.md"
    )
    runner.run()


if __name__ == "__main__":
    main()