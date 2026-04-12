# YAML loading with config merging
from pathlib import Path

import yaml


def load_yaml(path):
    """Load a YAML file into a flat dictionary."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base, override):
    """Merge two flat config dicts, with override taking precedence."""
    result = base.copy()
    result.update(override)
    return result


def coerce_override_value(value):
    """Convert CLI override values to simple Python scalars when possible."""
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def parse_overrides(opts):
    """Parse CLI overrides like ['method=retinexnet', 'batch_size=2'] into a dict."""
    overrides = {}
    for opt in opts:
        if "=" not in opt:
            raise ValueError(
                f"Invalid override '{opt}'. Expected key=value format, for example lr=0.0002."
            )
        key, value = opt.split("=", 1)
        overrides[key] = coerce_override_value(value)
    return overrides


def _load_component_defaults(config):
    """Load method, dataset, and metric defaults for the selected components."""
    defaults = {}

    if "method" in config:
        method_config_path = Path(f"configs/methods/{config['method']}.yaml")
        if method_config_path.exists():
            method_config = load_yaml(method_config_path)
            defaults = merge_configs(defaults, method_config)

    if "dataset" in config:
        dataset_config_path = Path(f"configs/datasets/{config['dataset']}.yaml")
        if dataset_config_path.exists():
            dataset_config = load_yaml(dataset_config_path)
            defaults = merge_configs(defaults, dataset_config)

    if "metrics" in config:
        metrics_config_path = Path("configs/metrics/standard.yaml")
        if metrics_config_path.exists():
            metrics_config = load_yaml(metrics_config_path)
            defaults = merge_configs(defaults, metrics_config)

    return defaults


def load_config(path, overrides=None):
    """Load an experiment config, merge component defaults, then re-apply overrides."""
    experiment_config = load_yaml(path)
    overrides = overrides or {}

    # Apply selection overrides first so switched method/dataset defaults can be discovered.
    selection_config = merge_configs(experiment_config, overrides)
    defaults = _load_component_defaults(selection_config)

    # Defaults <- experiment <- CLI overrides.
    config = merge_configs(defaults, experiment_config)
    return merge_configs(config, overrides)
