import torch
from tqdm import tqdm
from core.logger import Logger

class Evaluator:
    """Owns the evaluation loop."""
    def __init__(self, model, device, log_dir):
        self.model = model.to(device)
        self.device = device
        self.logger = Logger(log_dir)

    def evaluate(self, dataloader, metrics, method_name, dataset_name):
        """Run the full evaluation loop."""
        # set model to evaluation mode (disables dropout, batchnorm etc.)
        self.model.eval()
        for metric in metrics:
            metric.reset()

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {method_name} on {dataset_name}")
            for batch in progress_bar:
                batch = [x.to(self.device) for x in batch]
                predictions = self.model.enhance(batch)

                for i in range(predictions.shape[0]):
                    for metric in metrics:
                        metric.compute(predictions[i], batch[1][i])

        # after loop — aggregate each metric and log the result
        for metric in metrics:
            result = metric.aggregate()
            metric_name = metric.__class__.__name__.replace("Metric", "").lower()
            self.logger.log(method_name, dataset_name, metric_name, result)

    