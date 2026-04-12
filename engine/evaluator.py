import torch
from tqdm import tqdm

from core.logger import Logger
from core.runtime import move_batch_to_device


class Evaluator:
    """Owns the evaluation loop."""

    def __init__(self, model, device, log_dir):
        self.model = model.to(device)
        self.device = device
        self.logger = Logger(log_dir)

    def evaluate(self, dataloader, metrics, method_name, dataset_name):
        """Run the full evaluation loop."""
        self.model.eval()
        for metric in metrics:
            metric.reset()

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {method_name} on {dataset_name}")
            for batch in progress_bar:
                batch = move_batch_to_device(batch, self.device)
                predictions = self.model.enhance(batch)
                if predictions.device != self.device:
                    predictions = predictions.to(self.device)

                for index in range(predictions.shape[0]):
                    for metric in metrics:
                        metric.compute(predictions[index], batch[1][index])

        for metric in metrics:
            result = metric.aggregate()
            metric_name = metric.__class__.__name__.replace("Metric", "").lower()
            self.logger.log(method_name, dataset_name, metric_name, result)
