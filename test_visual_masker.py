import torch

from models.visual_masker import VisualMasker


def main():
    # Use CPU for now.
    device = torch.device("cpu")

    # Create the model.
    model = VisualMasker().to(device)

    # Fake 64x64 cover and secret images.
    #
    # Shape:
    # [batch, channels, height, width]
    cover = torch.rand(2, 3, 64, 64).to(device)
    secret = torch.rand(2, 3, 64, 64).to(device)

    # Forward pass.
    encoded = model(cover, secret)

    print("Cover shape:  ", cover.shape)
    print("Secret shape: ", secret.shape)
    print("Encoded shape:", encoded.shape)

    # Check output range.
    print("Encoded min:", encoded.min().item())
    print("Encoded max:", encoded.max().item())


if __name__ == "__main__":
    main()