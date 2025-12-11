"""
Dataset loading and preparation for football logo classification.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(augment=True):
    """
    Get image transformations.

    Args:
        augment: If True, apply data augmentation for training

    Returns:
        torchvision.transforms.Compose object
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to data directory with league subdirectories
        batch_size: Batch size for dataloaders
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms(augment=False)
    )

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = get_transforms(augment=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader, full_dataset.classes
