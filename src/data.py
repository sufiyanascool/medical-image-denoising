from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {directory}. "
            "Check the path and upload/copy the dataset first."
        )
    return {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def patient_id_from_filename(filename):
    """For NIH filenames such as 00000001_000.png, return patient ID 00000001."""
    return Path(filename).stem.split("_")[0]


class PairedXrayDataset(Dataset):
    """Pairs noisy inputs and clean targets by exact filename."""

    def __init__(self, noisy_dir, clean_dir, image_size=256):
        noisy_files = list_images(noisy_dir)
        clean_files = list_images(clean_dir)
        common_names = sorted(set(noisy_files).intersection(clean_files))

        if not common_names:
            raise ValueError(
                "No matching image filenames were found between "
                f"{noisy_dir} and {clean_dir}."
            )

        self.pairs = [
            (noisy_files[name], clean_files[name], name) for name in common_names
        ]
        self.filenames = [item[2] for item in self.pairs]
        self.patient_ids = [patient_id_from_filename(name) for name in self.filenames]
        self.image_size = int(image_size)

        missing_clean = sorted(set(noisy_files) - set(clean_files))
        missing_noisy = sorted(set(clean_files) - set(noisy_files))
        print(
            f"Matched pairs: {len(self.pairs)} | "
            f"noisy without clean: {len(missing_clean)} | "
            f"clean without noisy: {len(missing_noisy)}"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        noisy_path, clean_path, _ = self.pairs[index]
        size = (self.image_size, self.image_size)

        with Image.open(noisy_path) as image:
            noisy = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            noisy = np.asarray(noisy, dtype=np.float32) / 255.0

        with Image.open(clean_path) as image:
            clean = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            clean = np.asarray(clean, dtype=np.float32) / 255.0

        noisy_tensor = torch.from_numpy(noisy.copy()).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean.copy()).unsqueeze(0)
        return noisy_tensor, clean_tensor


def make_patient_splits(dataset, validation_fraction=0.15, test_fraction=0.15, seed=42):
    """Split by patient ID so images from the same patient do not cross partitions."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be greater than zero.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fractions must be less than 1.")

    patient_ids = sorted(set(dataset.patient_ids))
    if len(patient_ids) < 3:
        raise ValueError(
            "At least 3 distinct patient IDs are needed for patient-level "
            "train/validation/test splitting. Check NIH-style filenames."
        )

    random.Random(seed).shuffle(patient_ids)
    n_patients = len(patient_ids)
    n_test = max(1, round(n_patients * test_fraction))
    n_val = max(1, round(n_patients * validation_fraction))

    if n_test + n_val >= n_patients:
        n_test, n_val = 1, 1

    test_patients = set(patient_ids[:n_test])
    val_patients = set(patient_ids[n_test:n_test + n_val])
    train_patients = set(patient_ids[n_test + n_val:])

    groups = {
        "train": train_patients,
        "validation": val_patients,
        "test": test_patients,
    }
    indices = {
        split: [
            idx for idx, patient_id in enumerate(dataset.patient_ids)
            if patient_id in allowed_patients
        ]
        for split, allowed_patients in groups.items()
    }

    for split, split_indices in indices.items():
        if not split_indices:
            raise ValueError(
                f"The {split} split is empty. Add more distinct patients or "
                "adjust the validation/test fractions."
            )

    manifest = {
        split: {
            "patient_ids": sorted(groups[split]),
            "filenames": [dataset.filenames[idx] for idx in indices[split]],
        }
        for split in groups
    }
    return indices, manifest
