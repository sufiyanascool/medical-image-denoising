import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import PairedXrayDataset, make_patient_splits
from src.gan import PatchDiscriminator, UNetGenerator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_l1(generator, loader, criterion, device):
    generator.eval()
    total_loss = 0.0
    total_images = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = generator(noisy)
            loss = criterion(prediction, clean)
            total_loss += loss.item() * noisy.size(0)
            total_images += noisy.size(0)
    return total_loss / max(total_images, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Pix2Pix-style conditional GAN for paired X-ray denoising."
    )
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--noisy-dir", default="data/lowdose_025")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/best_pix2pix_generator.pth")
    parser.add_argument("--split-file", default="splits/pix2pix_patient_split.json")
    args = parser.parse_args()

    if args.image_size != 256:
        raise ValueError("This U-Net architecture expects 256 x 256 images.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

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

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(dataset, split_indices["train"]), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, split_indices["validation"]), shuffle=False, **loader_kwargs
    )

    print(
        f"Pairs: {len(dataset)} | "
        f"train: {len(split_indices['train'])} | "
        f"validation: {len(split_indices['validation'])} | "
        f"test held out: {len(split_indices['test'])}"
    )
    print(f"Patient-wise split manifest saved to: {split_path}")

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss()
    reconstruction_criterion = nn.L1Loss()
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        running_g = 0.0
        running_d = 0.0
        seen = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            batch_count = noisy.size(0)

            # Update discriminator with real and generated image pairs.
            with torch.no_grad():
                fake_for_d = generator(noisy)
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(noisy, clean)
            fake_logits = discriminator(noisy, fake_for_d.detach())
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)
            d_loss = 0.5 * (
                adversarial_criterion(real_logits, real_targets)
                + adversarial_criterion(fake_logits, fake_targets)
            )
            d_loss.backward()
            d_optimizer.step()

            # Update generator. Freeze discriminator parameters for this step.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            g_optimizer.zero_grad(set_to_none=True)
            generated = generator(noisy)
            generated_logits = discriminator(noisy, generated)
            g_adversarial = adversarial_criterion(
                generated_logits, torch.ones_like(generated_logits)
            )
            g_reconstruction = reconstruction_criterion(generated, clean)
            g_loss = g_adversarial + args.lambda_l1 * g_reconstruction
            g_loss.backward()
            g_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            running_g += g_loss.item() * batch_count
            running_d += d_loss.item() * batch_count
            seen += batch_count

        avg_g = running_g / max(seen, 1)
        avg_d = running_d / max(seen, 1)
        val_l1 = validation_l1(
            generator, val_loader, reconstruction_criterion, device
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"G loss: {avg_g:.4f} | D loss: {avg_d:.4f} | "
            f"validation L1: {val_l1:.6f}"
        )

        # Select checkpoints by validation reconstruction error, not by test metrics.
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "epoch": epoch,
                    "validation_l1": val_l1,
                    "image_size": args.image_size,
                    "seed": args.seed,
                    "model_name": "Pix2PixStyleUNetGenerator",
                    "lambda_l1": args.lambda_l1,
                },
                checkpoint_path,
            )
            print(f"Saved best generator: {checkpoint_path}")

    print(f"Training complete. Best validation L1: {best_val_l1:.6f}")
    print("The test split was not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
