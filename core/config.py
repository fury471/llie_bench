# YAML loading with config merging
import yaml

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
    """Load a config file and merge with defaults."""
    default_config = {
        'method': 'default_method',
        'dataset': 'default_dataset',
        'metric': 'default_metric',
        'seed': 42,
    }
    user_config = load_yaml(path)
    return merge_configs(default_config, user_config)