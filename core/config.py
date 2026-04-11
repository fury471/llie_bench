# YAML loading with config merging
import yaml
from pathlib import Path

def load_yaml(path):
    """Load a YAML file and merge with defaults."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base, override):
    """Use update() to merge two configs, with override taking precedence."""
    result = base.copy()
    result.update(override)
    return result

def load_config(path):
    """Load and merge experiment, method, dataset and metrics configs."""
    # load experiment config first
    config = load_yaml(path)

    # auto-load and merge method config
    if "method" in config:
        method_config_path = f"configs/methods/{config['method']}.yaml"
        if Path(method_config_path).exists():
            method_config = load_yaml(method_config_path)
            config = merge_configs(config, method_config)

    # auto-load and merge dataset config
    if "dataset" in config:
        dataset_config_path = f"configs/datasets/{config['dataset']}.yaml"
        if Path(dataset_config_path).exists():
            dataset_config = load_yaml(dataset_config_path)
            config = merge_configs(config, dataset_config)

    # auto-load and merge metrics config
    if "metrics" in config:
        metrics_config_path = "configs/metrics/standard.yaml"
        if Path(metrics_config_path).exists():
            metrics_config = load_yaml(metrics_config_path)
            config = merge_configs(config, metrics_config)

    return config