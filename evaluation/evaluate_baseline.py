import sys
import os

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from skimage.metrics import structural_similarity as ssim

from models.visual_masker import VisualMasker
from models.hidden_insight import HiddenInsight
from models.noise import add_gaussian_noise


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32

# Use the checkpoint we just trained
CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "baseline_epoch_10.pt"
)

# Test all four noise levels from the project
NOISE_LEVELS = [0.03, 0.05, 0.07, 0.10]

# Number of test batches.
# 20 batches = 640 images.
TEST_BATCHES = 20


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

print("Loading CIFAR-10 test set...")

dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "datasets"),
    train=False,
    download=True,
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

print("Test images:", len(dataset))


# ============================================================
# MODELS
# ============================================================

masker = VisualMasker().to(device)
decoder = HiddenInsight().to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint:")

print(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

masker.load_state_dict(
    checkpoint["masker"]
)

decoder.load_state_dict(
    checkpoint["decoder"]
)

masker.eval()
decoder.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Training noise factor:",
    checkpoint["noise_factor"]
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(mse):
    """
    Calculate PSNR for images in [0, 1].
    """

    if mse <= 0:
        return float("inf")

    return 10.0 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim_batch(
    reconstructed,
    original
):
    """
    Calculate average SSIM over a batch.
    """

    reconstructed = reconstructed.detach().cpu()
    original = original.detach().cpu()

    scores = []

    for i in range(reconstructed.shape[0]):

        # CHW -> HWC
        rec = reconstructed[i].permute(
            1, 2, 0
        ).numpy()

        orig = original[i].permute(
            1, 2, 0
        ).numpy()

        score = ssim(
            orig,
            rec,
            channel_axis=2,
            data_range=1.0
        )

        scores.append(score)

    return sum(scores) / len(scores)


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting baseline evaluation...\n")

results = []


with torch.no_grad():

    for noise_factor in NOISE_LEVELS:

        total_mse = 0.0
        total_ssim = 0.0
        batches = 0

        print(
            f"Testing noise factor: {noise_factor}"
        )

        for step, (cover, _) in enumerate(dataloader):

            if step >= TEST_BATCHES:
                break

            cover = cover.to(device)

            # Create a different secret image
            permutation = torch.randperm(
                cover.size(0),
                device=device
            )

            secret = cover[permutation]

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            encoded = masker(
                cover,
                secret
            )

            # ------------------------------------------------
            # Add test noise
            # ------------------------------------------------

            noisy_encoded = add_gaussian_noise(
                encoded,
                noise_factor
            )

            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            reconstructed = decoder(
                noisy_encoded
            )

            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            mse = torch.mean(
                (reconstructed - secret) ** 2
            ).item()

            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            batch_ssim = calculate_ssim_batch(
                reconstructed,
                secret
            )

            total_mse += mse
            total_ssim += batch_ssim

            batches += 1

        # ----------------------------------------------------
        # Average metrics
        # ----------------------------------------------------

        average_mse = total_mse / batches
        average_ssim = total_ssim / batches

        psnr = calculate_psnr(
            average_mse
        )

        results.append(
            (
                noise_factor,
                average_mse,
                psnr,
                average_ssim
            )
        )

        print(
            f"Noise {noise_factor:.2f} | "
            f"MSE: {average_mse:.6f} | "
            f"PSNR: {psnr:.2f} dB | "
            f"SSIM: {average_ssim:.4f}"
        )

        print()


# ============================================================
# FINAL TABLE
# ============================================================

print("=" * 65)

print(
    "BASELINE RESULTS"
)

print("=" * 65)

print(
    f"{'Noise':<10}"
    f"{'MSE':<15}"
    f"{'PSNR (dB)':<15}"
    f"{'SSIM':<10}"
)

print("-" * 65)

for noise, mse, psnr, ssim_value in results:

    print(
        f"{noise:<10.2f}"
        f"{mse:<15.6f}"
        f"{psnr:<15.2f}"
        f"{ssim_value:<10.4f}"
    )

print("=" * 65)

print("\nEvaluation complete.")

