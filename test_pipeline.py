import torch

from models.visual_masker import VisualMasker
from models.hidden_insight import HiddenInsight
from models.noise import add_gaussian_noise


def main():
    device = torch.device("cpu")

    masker = VisualMasker().to(device)
    decoder = HiddenInsight().to(device)

    # Simulated cover and secret images.
    cover = torch.rand(2, 3, 64, 64).to(device)
    secret = torch.rand(2, 3, 64, 64).to(device)

    # -----------------------------------------
    # STEP 1: Encode
    # -----------------------------------------

    encoded = masker(cover, secret)

    print("Encoded:", encoded.shape)

    # -----------------------------------------
    # STEP 2: Add Gaussian noise
    # -----------------------------------------

    noise_factor = 0.05

    noisy_encoded = add_gaussian_noise(
        encoded,
        noise_factor
    )

    print("Noisy encoded:", noisy_encoded.shape)

    # -----------------------------------------
    # STEP 3: Decode
    # -----------------------------------------

    reconstructed = decoder(noisy_encoded)

    print("Reconstructed:", reconstructed.shape)

    # -----------------------------------------
    # STEP 4: Basic sanity checks
    # -----------------------------------------

    print(
        "Reconstructed min:",
        reconstructed.min().item()
    )

    print(
        "Reconstructed max:",
        reconstructed.max().item()
    )

    print(
        "Contains NaN:",
        torch.isnan(reconstructed).any().item()
    )


if __name__ == "__main__":
    main()