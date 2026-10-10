import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import PairedXrayDataset, make_patient_splits
from src.model import DenoiseCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_epoch_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(noisy)
            loss = criterion(prediction, clean)
            if training:
                loss.backward()
                optimizer.step()

        batch_items = noisy.size(0)
        total_loss += loss.item() * batch_items
        total_items += batch_items

    return total_loss / max(total_items, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the CNN X-ray denoising baseline.")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--noisy-dir", default="data/lowdose_025")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--split-file", default="splits/patient_split.json")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch-size, and image-size must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PairedXrayDataset(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        image_size=args.image_size,
    )
    split_indices, split_manifest = make_patient_splits(
        dataset,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    split_payload = {
        "seed": args.seed,
        "image_size": args.image_size,
        "clean_dir": str(args.clean_dir),
        "noisy_dir": str(args.noisy_dir),
        "splits": split_manifest,
    }
    split_path = Path(args.split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        Subset(dataset, split_indices["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, split_indices["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Pairs: {len(dataset)} | "
        f"train: {len(split_indices['train'])} | "
        f"validation: {len(split_indices['validation'])} | "
        f"test held out: {len(split_indices['test'])}"
    )
    print(f"Patient-wise split manifest saved to: {split_path}")

    model = DenoiseCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = mean_epoch_loss(model, train_loader, criterion, device, optimizer)
        val_loss = mean_epoch_loss(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_mse": val_loss,
                    "image_size": args.image_size,
                    "seed": args.seed,
                    "model_name": "DenoiseCNN",
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    print(f"Training complete. Best validation MSE: {best_val_loss:.6f}")
    print("The test split was not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
