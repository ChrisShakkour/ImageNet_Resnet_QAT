import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch


def build_imagenet_loaders(cfg):
    """
    Build PyTorch dataloaders for the ImageNet dataset using torchvision.datasets.ImageFolder.
    
    Args:
        cfg (dict): The full configuration dictionary loaded from config.yaml.
    
    Returns:
        train_loader, val_loader (DataLoader): PyTorch dataloaders.
    """

    data_dir = cfg["data"]["dataset_path"]
    image_size = cfg["data"].get("image_size", 224)
    batch_size = cfg["data"].get("batch_size", 256)
    num_workers = cfg["data"].get("num_workers", 8)

    # --- Define normalization ---
    normalize = transforms.Normalize(
        mean=cfg["data"]["normalize"]["mean"],
        std=cfg["data"]["normalize"]["std"]
    )

    # --- Data augmentation for training ---
    train_transforms = []
    if cfg["data"]["augmentation"].get("random_resized_crop", True):
        train_transforms.append(transforms.RandomResizedCrop(image_size))
    else:
        train_transforms.append(transforms.Resize(int(image_size * 1.15)))
        train_transforms.append(transforms.CenterCrop(image_size))
    
    if cfg["data"]["augmentation"].get("horizontal_flip", True):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if cfg["data"]["augmentation"].get("color_jitter", False):
        train_transforms.append(transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2))

    # Add final steps
    train_transforms += [
        transforms.ToTensor(),
        normalize
    ]

    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transforms.Compose(train_transforms)
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_transforms
    )

    # Optional: use a smaller subset for debugging
    if "subset_ratio" in cfg["data"]:
        ratio = cfg["data"]["subset_ratio"]
        train_len = int(len(train_dataset) * ratio)
        val_len = int(len(val_dataset) * ratio)
        train_dataset = Subset(train_dataset, range(train_len))
        val_dataset = Subset(val_dataset, range(val_len))
        print(f"⚠️ Using subset: {ratio*100:.1f}% of ImageNet for faster debugging")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✅ ImageNet loaders ready: "
          f"{len(train_dataset)} train images, {len(val_dataset)} val images")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage for testing
    import yaml

    with open("configs/resnet50.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader = build_imagenet_loaders(cfg)

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
