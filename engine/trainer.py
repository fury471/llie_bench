import torch
from tqdm import tqdm

from core.checkpoint import CheckpointManager
from core.logger import Logger
from core.runtime import move_batch_to_device
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
        set_seed(seed)
        use_amp = self.device == "cuda"
        amp_device = "cuda" if use_amp else "cpu"
        scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    loss = self.model.compute_loss(batch)

                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Non-finite loss encountered during training: {loss.detach().item()}"
                    )

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.detach().item()
                progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))

            avg_loss = epoch_loss / len(dataloader)
            self.logger.log("train", "train", "loss", avg_loss)
            self.ckpt_manager.save(self.model, self.optimizer, epoch)
