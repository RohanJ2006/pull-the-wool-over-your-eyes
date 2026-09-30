import sys
import os

# Make project root importable
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.visual_masker import VisualMasker
from models.hidden_insight import HiddenInsight
from models.noise import add_gaussian_noise

IMAGE_SIZE = 64
BATCH_SIZE = 32

EPOCHS = 10

STEPS_PER_EPOCH = 100

LEARNING_RATE = 0.001
NOISE_FACTOR = 0.05

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

print("Loading CIFAR-10...")

dataset = datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "datasets"),
    train=True,
    download=True,
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

print("Dataset size:", len(dataset))

masker = VisualMasker().to(device)
decoder = HiddenInsight().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    list(masker.parameters()) +
    list(decoder.parameters()),
    lr=LEARNING_RATE,
)

print("\nStarting baseline training...\n")

for epoch in range(EPOCHS):

    masker.train()
    decoder.train()

    total_loss = 0.0

    progress = tqdm(
        dataloader,
        total=STEPS_PER_EPOCH,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for step, (cover, _) in enumerate(progress):

        if step >= STEPS_PER_EPOCH:
            break
            
        cover = cover.to(device)

        permutation = torch.randperm(
            cover.size(0),
            device=device
        )

        secret = cover[permutation]

        encoded = masker(
            cover,
            secret
        )

        noisy_encoded = add_gaussian_noise(
            encoded,
            NOISE_FACTOR
        )

        reconstructed = decoder(
            noisy_encoded
        )

        loss = criterion(
            reconstructed,
            secret
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    average_loss = total_loss / min(
        STEPS_PER_EPOCH,
        len(dataloader)
    )

    print(
        f"Epoch {epoch + 1}: "
        f"loss = {average_loss:.6f}"
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"baseline_epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "masker": masker.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": average_loss,
            "noise_factor": NOISE_FACTOR,
        },
        checkpoint_path,
    )

    print(
        "Saved:",
        checkpoint_path
    )


print("\nBaseline training complete.")
