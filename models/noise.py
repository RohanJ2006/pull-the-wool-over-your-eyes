import torch


def add_gaussian_noise(image, noise_factor):
    """
    Add Gaussian noise to an image.

    Args:
        image:
            Tensor with shape [B, C, H, W].
            Expected range: [0, 1]

        noise_factor:
            Standard deviation of the Gaussian noise.

    Returns:
        Noisy image clipped to [0, 1].
    """

    noise = torch.randn_like(image) * noise_factor

    noisy_image = image + noise

    return torch.clamp(noisy_image, 0.0, 1.0)