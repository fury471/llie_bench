from pathlib import Path
import warnings


class Protocol:
    """Protocol checks that keep benchmark runs compatible and reproducible."""

    def __init__(self, protocol_path):
        self.protocol_path = Path(protocol_path)
        self.rules = self.protocol_path.read_text(encoding="utf-8")

    def check_compatibility(self, method_meta, dataset_meta):
        """Check whether a method and dataset are compatible for evaluation."""
        method_type = method_meta.get("type")
        dataset_type = dataset_meta.get("type")
        if method_type != dataset_type:
            raise ValueError(
                f"Method type '{method_type}' is not compatible with dataset type '{dataset_type}'."
            )

    def check_seed(self, config):
        """Warn when a config omits a seed value."""
        if "seed" not in config:
            warnings.warn("No seed set in config. Results may not be reproducible.")

    def check_norm(self, tensor):
        """Warn when a tensor is not normalized to the [0, 1] range."""
        if tensor.min() < 0 or tensor.max() > 1:
            warnings.warn("Tensor values are not normalized to [0, 1].")
