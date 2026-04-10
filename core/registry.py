class Registry:
    """A registry that supports both decorator and manual registration."""

    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name=None):
        """Decorator to register a class."""
        def decorator(cls):
            key = name if name else cls.__name__.lower()
            self._registry[key] = cls
            return cls
        return decorator

    def __getitem__(self, name):
        if name not in self._registry:
            raise ValueError(f"'{name}' not found in {self._name}. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __setitem__(self, name, cls):
        self._registry[name] = cls

    def keys(self):
        return self._registry.keys()


# Global registries
METHOD_REGISTRY = Registry("METHOD_REGISTRY")
DATASET_REGISTRY = Registry("DATASET_REGISTRY")
METRIC_REGISTRY = Registry("METRIC_REGISTRY")


def lookup(registry, name):
    """Lookup a class from a registry by name."""
    if name not in registry:
        raise ValueError(f"'{name}' not found. Available: {list(registry.keys())}")
    return registry[name]