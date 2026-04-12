from torch.utils.data import DataLoader

from core.logger import Logger
from core.protocol import Protocol
from engine.evaluator import Evaluator


class BenchmarkRunner:
    """Owns the full benchmark loop."""

    def __init__(self, methods, datasets, metrics, device, log_dir, protocol_path, batch_size=1):
        self.methods = methods
        self.datasets = datasets
        self.metrics = metrics
        self.device = device
        self.batch_size = batch_size
        self.logger = Logger(log_dir)
        self.protocol = Protocol(protocol_path)

    def run(self):
        """Run the full benchmark loop."""
        for method in self.methods:
            method_name = method.get_meta()["name"]
            for dataset in self.datasets:
                dataset_name = dataset.get_meta()["name"]
                self.protocol.check_compatibility(method.get_meta(), dataset.get_meta())

                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                evaluator = Evaluator(method, self.device, self.logger.log_dir / "eval")
                print(f"Running {method_name} on {dataset_name}...")
                evaluator.evaluate(dataloader, self.metrics, method_name, dataset_name)
