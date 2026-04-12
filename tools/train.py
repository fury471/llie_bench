import argparse

import torch

import plugins
from core.config import load_config, parse_overrides
from core.runtime import load_method_checkpoint, method_requires_checkpoint, resolve_device
from core.transforms import build_transforms
from core.registry import DATASET_REGISTRY, METHOD_REGISTRY, lookup
from engine.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train an LLIE method")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--opts", nargs="+", default=[], help="Override config values like lr=0.0002")
    args = parser.parse_args()

    try:
        overrides = parse_overrides(args.opts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = load_config(args.config, overrides=overrides)
    transforms = build_transforms(config)

    method = lookup(METHOD_REGISTRY, config["method"])()
    if not method_requires_checkpoint(method):
        print(f"Method '{config['method']}' is a traditional method and does not support training.")
        return

    if hasattr(method, "set_phase"):
        phase = config.get("train_phase", "decom")
        method.set_phase(phase)
        print(f"Training phase: {phase}")

    if config.get("ckpt"):
        try:
            _, status = load_method_checkpoint(
                method,
                config["method"],
                ckpt=config.get("ckpt"),
                required=False,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(status)

    dataset = lookup(DATASET_REGISTRY, config["dataset"])(
        config["data_root"],
        split="train",
        transforms=transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, method.parameters()),
        lr=config["lr"],
    )

    device = resolve_device()
    print(f"Using device: {device}")

    trainer = Trainer(
        model=method,
        optimizer=optimizer,
        device=device,
        log_dir=config["log_dir"],
        ckpt_dir=config["ckpt_dir"],
    )
    trainer.train(
        dataloader=train_loader,
        epochs=config["epochs"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    main()
