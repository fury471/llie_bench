import torch
from pathlib import Path

class CheckpointManager:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, epoch):
        """Save model and optimizer state to a checkpoint file."""
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"   
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def load(self, path):
        """Load model and optimizer state from a checkpoint file."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path)
        print(f"Checkpoint loaded: {ckpt_path} (epoch {checkpoint['epoch']})")
        return checkpoint


def resolve_checkpoint_path(ckpt=None, ckpt_dir=None):
    """Return an explicit checkpoint path or the latest checkpoint in a directory."""
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if ckpt_dir:
        ckpt_dir_path = Path(ckpt_dir)
        if not ckpt_dir_path.exists():
            return None
        ckpts = sorted(ckpt_dir_path.glob("*.pth"))
        if ckpts:
            return ckpts[-1]

    return None
