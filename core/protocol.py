from pathlib import Path
import warnings
import torch

class Protocol:
    """Protocol class to manage the overall workflow of training and evaluation."""
    def __init__(self, protocol_path):
        self.protocol_path = Path(protocol_path)
        self.rules = Path(protocol_path).read_text()
    
    def check_compatibility(self, method_meta, dataset_meta):
        """Check if the method and dataset are compatible according to the protocol rules."""
        method_type = method_meta.get("type")
        dataset_type = dataset_meta.get("type")
        
        if method_type != dataset_type:
            raise ValueError(f"Method type '{method_type}' is not compatible with dataset type '{dataset_type}'.")
        
    def check_seed(self, config):
        """Should have a "seed" key in the config for reproducibility."""
        if "seed" not in config:
            warnings.warn("No seed set in config. Results may not be reproducible.")

    def check_norm(self, tensor):
        """Check if the tensor is normalized (values between 0 and 1)."""
        if tensor.min() < 0 or tensor.max() > 1:
            warnings.warn("Tensor values are not normalized to [0, 1].")