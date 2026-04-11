import argparse
from core.config import load_config
from core.registry import METHOD_REGISTRY, DATASET_REGISTRY, lookup
from core.transforms import RandomCrop, RandomFlip, Compose
from engine.trainer import Trainer
import torch
from pathlib import Path

# import plugins so they register themselves
import plugins

# all the logic here
def main():
    # parse the --config argument.
    parser = argparse.ArgumentParser(description="Train LLIE method")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--opts", nargs="+", default=[], help="Override config values e.g. method=clahe lr=0.0002")
    args = parser.parse_args()

    # load the config and merge with defaults
    config = load_config(args.config)
    for opt in args.opts:
        key, value = opt.split("=")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        config[key] = value

    # build transforms from config
    transform_list = []
    if config.get("patch_size"):
        transform_list.append(RandomCrop(config["patch_size"]))
    if config.get("random_flip", False):
        transform_list.append(RandomFlip())
    transforms = Compose(transform_list) if transform_list else None

    # lookup the method and dataset classes
    method = lookup(METHOD_REGISTRY, config["method"])()
    # check if method supports training
    if not list(method.parameters()):
        print(f"Method '{config['method']}' is a traditional method and does not support training.")
        return
    dataset = lookup(DATASET_REGISTRY, config["dataset"])(
        config["data_root"],
        split="train",
        transforms=transforms
    )

    # create a dataloader for training
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    # in tools/train.py, after creating method:
    if hasattr(method, 'set_phase'):
        phase = config.get('train_phase', 'decom')
        method.set_phase(phase)
        print(f"Training phase: {phase}")

    # load checkpoint if specified (required for Phase 2)
    if config.get("ckpt"):
        method.load_ckpt(config["ckpt"])
        print(f"Loaded checkpoint: {config['ckpt']}")

    # only optimize parameters that require gradients
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, method.parameters()),
        lr=config["lr"]
    )

    # detect the device automatically (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # create the trainer and run training
    trainer = Trainer(
        model=method,
        optimizer=optimizer,
        device=device,
        log_dir=config["log_dir"],
        ckpt_dir=config["ckpt_dir"]
    )
    trainer.train(
        dataloader=train_loader,
        epochs=config["epochs"],
        seed=config["seed"]
    )


if __name__ == "__main__": 
    main()