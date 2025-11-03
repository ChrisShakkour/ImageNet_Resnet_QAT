import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def build_fake_loaders(cfg):
    """
    Build PyTorch dataloaders using torchvision.datasets.FakeData.
    This simulates a dataset with random images and labels.
    
    Args:
        cfg (dict): Configuration dictionary loaded from config.yaml.
    
    Returns:
        train_loader, val_loader (DataLoader)
    """

    # --- Extract data configuration ---
    # `cfg` is expected to provide a `dataloader` namespace (e.g. Munch or object)
    # Access attributes directly and fall back to sensible defaults when missing.
    num_classes = cfg.dataloader.num_classes
    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.workers
    image_size = cfg.dataloader.image_size
    channels = cfg.dataloader.channels
    train_size = cfg.dataloader.train_size
    val_size = cfg.dataloader.val_size

    # --- Transforms ---
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # --- Fake datasets ---
    train_dataset = datasets.FakeData(
        size=train_size,
        image_size=(channels, image_size, image_size),
        num_classes=num_classes,
        transform=transform
    )

    val_dataset = datasets.FakeData(
        size=val_size,
        image_size=(channels, image_size, image_size),
        num_classes=num_classes,
        transform=transform
    )

    # --- Data loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✅ FakeData loaders ready: {train_size} train, {val_size} val, "
          f"{num_classes} classes, image size {image_size}x{image_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example standalone test
    import yaml

    # You can reuse the same config structure as ImageNet or CIFAR
    example_cfg = {
        "data": {
            "num_classes": 1000,
            "batch_size": 256,
            "num_workers": 4,
            "image_size": 224,
            "train_size": 5000,
            "val_size": 1000
        }
    }

    train_loader, val_loader = build_fake_loaders(example_cfg)

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")

    images = [images[i] for i in range(4)]
    grid = make_grid(images, nrow=2)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()