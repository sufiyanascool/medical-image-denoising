# AI-Based Medical Image Enhancement for Chest X-rays

A PyTorch project exploring denoising of synthetically degraded chest X-ray images. The repository includes a lightweight CNN baseline and a Pix2Pix-style conditional GAN experiment.

## Project objective

Explore whether a neural network can reduce simulated noise while preserving image structure. This is an educational/research proof of concept. It is **not clinically validated** and must not be used for diagnosis.

## Dataset and original prototype

- **Dataset:** NIH ChestX-ray14.
- **Images used in the original Google Colab prototype:** 407 chest X-rays.
- **Reference images:** original images were treated as clean targets.
- **Degradation:** synthetic Poisson-based intensity corruption with settings named `lowdose_010` and `lowdose_025`.
- **Preprocessing:** grayscale images resized to 256 × 256.
- **Original CNN training:** PyTorch, MSE loss, Adam optimizer (learning rate 0.001), batch size 8, 10 epochs.

The dataset is not included in this repository. Obtain it through its official distribution and follow its terms of use. Do not commit medical images or model checkpoints to this repository.

### Important note about the noise simulation

The lowdose_010 and lowdose_025 names refer to simulation parameters, not measured 10% or 25% clinical X-ray exposures. Synthetic corruption alone does not establish real-world dose equivalence.

The original Colab prototype used a simple Poisson sampling procedure. The included data-generation script is a cleaned-up, mean-preserving intensity-domain Poisson simulation for reproducible experiments; its results should not be assumed to reproduce the earlier notebook exactly or to model a specific scanner's physics.

## Models

### 1. CNN baseline

A custom CNN encoder-decoder, not a pretrained ResNet/DenseNet backbone:

1. Conv2D: 1 → 32 channels, ReLU
2. Conv2D: 32 → 64 channels, ReLU
3. Max pooling
4. Transposed convolution: 64 → 32 channels, ReLU
5. Conv2D: 32 → 1 channel, Sigmoid

Files: `src/model.py`, `train.py`, `evaluate.py`.

### 2. Pix2Pix-style conditional GAN

The GAN experiment uses:
- **Generator:** U-Net-style encoder-decoder with skip connections.
- **Discriminator:** conditional PatchGAN that receives a noisy image concatenated with either the clean target or generated output.
- **Generator objective:** adversarial BCE loss plus an L1 reconstruction loss. The default L1 weight is 100.
- **Optimizers:** Adam, learning rate 0.0002, betas (0.5, 0.999).
- **Checkpoint selection:** lowest validation L1 loss, not test-set results.

This GAN implementation is a research baseline. It has not been shown to outperform the CNN, and the results must be measured on the same held-out split before drawing conclusions.

Files: `src/gan.py`, `train_gan.py`, `evaluate_gan.py`.

## Repository layout

- `src/model.py` — CNN encoder-decoder.
- `src/gan.py` — U-Net generator and PatchGAN discriminator.
- `src/data.py` — paired image loading and patient-level split helpers.
- `scripts/generate_poisson_noise.py` — create synthetic noisy inputs from clean images.
- `train.py` / `evaluate.py` — CNN training and held-out evaluation.
- `train_gan.py` / `evaluate_gan.py` — GAN training and held-out evaluation.
- `requirements.txt` — Python dependencies.

## Setup

Install dependencies:

    pip install -r requirements.txt

Place reference images into `data/clean/`. The split utility expects NIH-style filenames such as `00000001_000.png`, where the prefix identifies the patient. If using a differently named dataset, adapt and verify the patient-ID parsing before relying on the patient-level split.

Generate noisy images:

    python scripts/generate_poisson_noise.py --clean-dir data/clean --output-dir data/lowdose_025 --dose-fraction 0.25 --seed 42

Train the CNN baseline:

    python train.py --clean-dir data/clean --noisy-dir data/lowdose_025 --epochs 10

Evaluate the CNN on its held-out patient groups:

    python evaluate.py --clean-dir data/clean --noisy-dir data/lowdose_025 --checkpoint checkpoints/best_model.pth --split-file splits/patient_split.json

Train the Pix2Pix-style GAN (start with a small number of epochs to test GPU memory/runtime):

    python train_gan.py --clean-dir data/clean --noisy-dir data/lowdose_025 --epochs 30 --batch-size 4

Evaluate the GAN on its held-out patient groups:

    python evaluate_gan.py --clean-dir data/clean --noisy-dir data/lowdose_025 --checkpoint checkpoints/best_pix2pix_generator.pth --split-file splits/pix2pix_patient_split.json

The training scripts save separate patient-wise split manifests. Preserve the generated manifests with experiment results. Do not evaluate on training images and describe those metrics as held-out test performance.

## Preliminary results from the original Colab prototype

The earlier CNN experiment reported a best validation MSE of approximately **0.000365**. An exploratory calculation on 50 samples reported **33.68 dB PSNR** and **0.9318 SSIM**.

These are preliminary prototype results, not independent test-set scores. The earlier evaluation did not establish that all 50 images came from a patient-disjoint held-out test set, so these figures should not be presented as final generalization performance. No CNN-versus-GAN result is claimed until both models have been trained and evaluated with a documented, comparable protocol.

## Recommended next steps

1. Run and verify the CNN baseline with patient-level train/validation/test separation.
2. Train the GAN and compare both models on the same held-out patient groups.
3. Report noisy-input and denoised-output PSNR/SSIM, plus qualitative examples and residual maps.
4. Inspect image details for loss or hallucination of anatomical structures.
5. If making claims about actual low-dose imaging, validate the degradation model against suitable real low-dose data or an established imaging simulation.

## Limitations

- The current experiments use synthetic corruption, not measured low-dose acquisitions.
- Pixel metrics and visual realism alone do not demonstrate diagnostic preservation, clinical utility, or safety.
- GANs may introduce plausible-looking details that are unsupported by the input.
- This project is for research and educational exploration only.
