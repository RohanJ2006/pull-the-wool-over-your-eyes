import torch


def add_gaussian_noise(image, noise_factor):

    noise = torch.randn_like(image) * noise_factor

    noisy_image = image + noise

    return torch.clamp(noisy_image, 0.0, 1.0)