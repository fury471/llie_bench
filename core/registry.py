# Lookup table connecting string names to classes
METHOD_REGISTRY = {}
DATASET_REGISTRY = {}
METRIC_REGISTRY = {}

def lookup(registry, name):
    """Lookup a class from a registry by name."""
    if name not in registry:
        raise ValueError(f"'{name}' not found. Available: {list(registry.keys())}")
    return registry[name]