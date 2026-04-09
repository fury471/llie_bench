import torch
from tqdm import tqdm
from core.checkpoint import CheckpointManager
from core.logger import Logger
from core.seed import set_seed

class Trainer:
    """Owns the training loop."""
    def __init__(self, model, optimizer, device, log_dir, ckpt_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.logger = Logger(log_dir)
        self.ckpt_manager = CheckpointManager(ckpt_dir)

    def train(self, dataloader, epochs, seed):
        """Run the full training loop."""
        # set seed before anything touches data or model
        set_seed(seed)

        for epoch in range(epochs):
            # set model to training mode (enables dropout, batchnorm etc.)
            self.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                # move batch to the correct device (GPU or CPU)
                batch = [x.to(self.device) for x in batch]

                # step 1: forward pass — model sees the batch
                # step 2: compute loss — measure how wrong the predictions are
                loss = self.model.compute_loss(batch)

                # step 3: zero gradients — clear old gradients from previous step
                self.optimizer.zero_grad()

                # step 4: backward pass — compute new gradients
                loss.backward()

                # step 5: update weights — optimizer improves the model
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))

            # after each epoch — log average loss and save checkpoint
            avg_loss = epoch_loss / len(dataloader)
            self.logger.log("train", "train", "loss", avg_loss)
            self.ckpt_manager.save(self.model, self.optimizer, epoch)