from engine.evaluator import Evaluator
from core.protocol import Protocol
from core.logger import Logger
from torch.utils.data import DataLoader

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
        # for each method:
        #     for each dataset:
        #         evaluate method on dataset
        for method in self.methods:
            for dataset in self.datasets:
                # check protocol to see if this method-dataset pair should be evaluated
                self.protocol.check_compatibility(method.get_meta(), dataset.get_meta())

                evaluator = Evaluator(method, self.device, self.logger.log_dir / "eval")
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                method_name = method.get_meta()["name"]
                dataset_name = dataset.get_meta()["name"]
                print(f"Running {method_name} on {dataset_name}...")
                evaluator.evaluate(dataloader, self.metrics, method_name, dataset_name)
