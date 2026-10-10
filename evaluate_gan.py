import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Subset

from src.data import PairedXrayDataset
from src.gan import UNetGenerator


def compute_metrics(reference, prediction):
    reference = np.clip(reference, 0.0, 1.0)
    prediction = np.clip(prediction, 0.0, 1.0)
    return {
        "psnr_db": float(peak_signal_noise_ratio(reference, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(reference, prediction, data_range=1.0)),
    }


def save_triplet(noisy, restored, clean, output_path):
    images = [
        Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))
        for array in (noisy, restored, clean)
    ]
    canvas = Image.new("L", (images[0].width * 3, images[0].height))
    for index, image in enumerate(images):
        canvas.paste(image, (index * image.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pix2Pix-style generator on held-out patients."
    )
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--noisy-dir", default="data/lowdose_025")
    parser.add_argument("--checkpoint", default="checkpoints/best_pix2pix_generator.pth")
    parser.add_argument("--split-file", default="splits/pix2pix_patient_split.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report", default="outputs/pix2pix_test_metrics.json")
    parser.add_argument("--preview-dir", default="outputs/pix2pix_previews")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    split_path = Path(args.split_file)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the GAN first.")
    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_path}. Train the GAN first.")

    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    test_filenames = manifest["splits"]["test"]["filenames"]
    image_size = int(manifest.get("image_size", 256))
    if image_size != 256:
        raise ValueError("This U-Net architecture expects 256 x 256 images.")

    dataset = PairedXrayDataset(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        image_size=image_size,
    )
    index_by_filename = {name: index for index, name in enumerate(dataset.filenames)}
    missing = sorted(set(test_filenames) - set(index_by_filename))
    if missing:
        raise ValueError(
            "Some test files from the split manifest are missing: "
            + ", ".join(missing[:10])
        )
    test_indices = [index_by_filename[name] for name in test_filenames]
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = UNetGenerator().to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    noisy_metrics = []
    denoised_metrics = []
    preview_dir = Path(args.preview_dir)
    preview_count = 0

    with torch.no_grad():
        for noisy_batch, clean_batch in test_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            restored_batch = generator(noisy_batch)

            for index in range(noisy_batch.size(0)):
                noisy = noisy_batch[index, 0].cpu().numpy()
                clean = clean_batch[index, 0].cpu().numpy()
                restored = restored_batch[index, 0].cpu().numpy()
                noisy_metrics.append(compute_metrics(clean, noisy))
                denoised_metrics.append(compute_metrics(clean, restored))

                if preview_count < 12:
                    filename = test_filenames[preview_count]
                    save_triplet(
                        noisy,
                        restored,
                        clean,
                        preview_dir / f"{Path(filename).stem}_noisy-denoised-clean.png",
                    )
                    preview_count += 1

    report = {
        "checkpoint": str(checkpoint_path),
        "test_images": len(test_indices),
        "test_patients": len(manifest["splits"]["test"]["patient_ids"]),
        "noisy_input_vs_clean": {
            "mean_psnr_db": float(np.mean([item["psnr_db"] for item in noisy_metrics])),
            "mean_ssim": float(np.mean([item["ssim"] for item in noisy_metrics])),
        },
        "denoised_output_vs_clean": {
            "mean_psnr_db": float(np.mean([item["psnr_db"] for item in denoised_metrics])),
            "mean_ssim": float(np.mean([item["ssim"] for item in denoised_metrics])),
        },
        "note": (
            "Metrics are computed on the manifest's held-out patient split. "
            "Synthetic noise metrics do not establish clinical validity."
        ),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics: {report_path}")
    print(f"Saved up to 12 previews to: {preview_dir}")


if __name__ == "__main__":
    main()
