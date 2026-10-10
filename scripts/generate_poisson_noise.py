import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def simulate_poisson_corruption(image_float, dose_fraction=0.25, scale=10000, rng=None):
    """
    Apply a simple mean-preserving Poisson intensity simulation to a [0, 1] image.

    dose_fraction is a synthetic noise-strength parameter; it is not a calibrated
    estimate of real clinical X-ray exposure or scanner physics.
    """
    if not 0 < dose_fraction <= 1:
        raise ValueError("dose_fraction must be in the interval (0, 1].")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    image_float = np.clip(image_float.astype(np.float32), 0.0, 1.0)
    expected_counts = image_float * float(scale) * float(dose_fraction)
    sampled_counts = rng.poisson(expected_counts)
    corrupted = sampled_counts.astype(np.float32) / (float(scale) * float(dose_fraction))
    return np.clip(corrupted, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Poisson-corrupted images from clean grayscale images."
    )
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--output-dir", default="data/lowdose_025")
    parser.add_argument("--dose-fraction", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean-image directory not found: {clean_dir}")

    image_paths = sorted(
        path for path in clean_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in {clean_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("L"), dtype=np.float32) / 255.0

        corrupted = simulate_poisson_corruption(
            image,
            dose_fraction=args.dose_fraction,
            scale=args.scale,
            rng=rng,
        )
        output_image = Image.fromarray(np.round(corrupted * 255.0).astype(np.uint8))
        output_image.save(output_dir / image_path.name)

    print(f"Generated {len(image_paths)} synthetic images in {output_dir}")
    print(
        "Note: this simplified simulation parameter does not establish "
        "equivalence to a measured clinical X-ray dose."
    )


if __name__ == "__main__":
    main()
