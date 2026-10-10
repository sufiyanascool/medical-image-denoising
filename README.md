# AI-Based Medical Image Enhancement for Chest X-rays

A PyTorch project exploring CNN-based denoising of synthetically degraded chest X-ray images. The current baseline is a lightweight CNN encoder-decoder. A Pix2Pix-style conditional GAN is a planned extension, not yet implemented.

## Project objective

Study whether a neural network can reduce simulated image noise while preserving the structure of chest X-rays. This is an educational/research proof of concept. It is **not clinically validated** and must not be used for diagnosis.

## Dataset and original prototype

- **Dataset:** NIH ChestX-ray14.
- **Images used in the original Google Colab prototype:** 407 chest X-rays.
- **Reference images:** original images were treated as clean targets.
- **Degradation:** synthetic Poisson-based intensity corruption with settings named lowdose_010 and lowdose_025.
- **Preprocessing:** grayscale images resized to 256 × 256.
- **Baseline training:** PyTorch, MSE loss, Adam optimizer (learning rate 0.001), batch size 8, 10 epochs.

The dataset is not included in this repository. Obtain it through its official distribution and follow its terms of use. Do not commit medical images or model checkpoints to this repository.

### Important note about the noise simulation

The lowdose_010 and lowdose_025 names refer to simulation parameters, not measured 10% or 25% clinical X-ray exposures. Synthetic corruption alone does not establish real-world dose equivalence.

The original Colab prototype used a simple Poisson sampling procedure. The included data-generation script is a cleaned-up, mean-preserving intensity-domain Poisson simulation for reproducible experiments; its results should not be assumed to reproduce the earlier notebook exactly or to model a specific scanner's physics.

## Baseline architecture

The CNN is a custom encoder-decoder, not a pretrained ResNet/DenseNet backbone:

1. Conv2D: 1 → 32 channels, ReLU
2. Conv2D: 32 → 64 channels, ReLU
3. Max pooling
4. Transposed convolution: 64 → 32 channels, ReLU
5. Conv2D: 32 → 1 channel, Sigmoid

The generator outputs a denoised grayscale image in the range [0, 1].

## Repository layout

- src/model.py — CNN encoder-decoder architecture.
- src/data.py — paired image loading and patient-level train/validation/test split helpers.
- scripts/generate_poisson_noise.py — create synthetic noisy inputs from clean images.
- train.py — train the baseline and save the patient-wise split manifest.
- evaluate.py — evaluate noisy and denoised images against clean references on the held-out test partition.
- requirements.txt — Python dependencies.

## Setup

Create a virtual environment if desired, then install dependencies:

    pip install -r requirements.txt

Place your reference images in a local data directory. The NIH ChestX-ray14 filename format (for example, 00000001_000.png) allows images from the same patient to be grouped for splitting.

Generate noisy images using one fixed simulation setting:

    python scripts/generate_poisson_noise.py --clean-dir data/clean --output-dir data/lowdose_025 --dose-fraction 0.25 --seed 42

Train the CNN:

    python train.py --clean-dir data/clean --noisy-dir data/lowdose_025 --epochs 10

Evaluate only the held-out patient groups produced by training:

    python evaluate.py --clean-dir data/clean --noisy-dir data/lowdose_025 --checkpoint checkpoints/best_model.pth --split-file splits/patient_split.json

The first run of train.py writes splits/patient_split.json. Keep this manifest with experiment records so that evaluation can be reproduced. Do not evaluate on all images and describe that result as held-out test performance.

## Preliminary results from the original Colab prototype

The earlier experiment reported a best validation MSE of approximately **0.000365**. An exploratory calculation on 50 samples reported **33.68 dB PSNR** and **0.9318 SSIM**.

These are preliminary prototype results, not independent test-set scores. The earlier evaluation did not establish that all 50 images came from a patient-disjoint held-out test set, so the figures should not be presented as final generalization performance or directly compared with a new run without matching the evaluation protocol.

## Planned next steps

1. Reproduce the CNN baseline with patient-level train/validation/test separation.
2. Compare noisy input and denoised output on the held-out test patients using PSNR and SSIM.
3. Inspect visual examples and residual maps for lost or invented anatomical detail.
4. Implement a Pix2Pix-style conditional GAN with a U-Net generator and PatchGAN discriminator, then compare it with the CNN baseline using the same splits and metrics.
5. If making claims about actual low-dose imaging, validate the degradation model against suitable real low-dose data or an established imaging simulation.

## Limitations

- No measured low-dose acquisitions are included in this initial experiment.
- Image-level pixel metrics alone do not demonstrate diagnostic preservation, clinical utility, or safety.
- GANs can produce visually plausible details that are not supported by the input image.
- This project is for research and educational exploration only.
