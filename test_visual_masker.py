import torch

from models.visual_masker import VisualMasker


def main():
    device = torch.device("cpu")

    model = VisualMasker().to(device)

    cover = torch.rand(2, 3, 64, 64).to(device)
    secret = torch.rand(2, 3, 64, 64).to(device)

    encoded = model(cover, secret)

    print("Cover shape:  ", cover.shape)
    print("Secret shape: ", secret.shape)
    print("Encoded shape:", encoded.shape)

    print("Encoded min:", encoded.min().item())
    print("Encoded max:", encoded.max().item())

if __name__ == "__main__":
    main()