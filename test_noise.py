import torch

from models.noise import add_gaussian_noise


def main():
    image = torch.rand(2, 3, 64, 64)

    for noise_factor in [0.03, 0.05, 0.07, 0.10]:
        noisy = add_gaussian_noise(image, noise_factor)

        difference = torch.mean(
            torch.abs(noisy - image)
        )

        print(
            f"Noise factor: {noise_factor:.2f} | "
            f"Mean difference: {difference.item():.6f}"
        )


if __name__ == "__main__":
    main()