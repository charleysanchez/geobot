from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
from PIL import Image
import csv
import os
import random


class ImageToGeoDataset(Dataset):
    """Dataset for mapping Street View images to geographic coordinates."""

    def __init__(
        self,
        root="dataset",
        split=None,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        img_transform=None,
        normalize_coords=True
    ):
        """
        Args:
            root: Path to dataset directory containing 'images' and 'coords' subdirs
            split: One of "train", "val", "test", or None (full dataset)
            train_ratio: Fraction of data for training (default 0.8)
            val_ratio: Fraction of data for validation (default 0.1)
            seed: Random seed for reproducible splits
            img_transform: Optional transform for images
            normalize_coords: Whether to normalize coords to [-1, 1] range
        """
        self.img_dir = os.path.join(root, "images")
        self.coords_dir = os.path.join(root, "coords")
        self.normalize_coords = normalize_coords

        # Load all image files
        all_img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ], key=lambda x: int(os.path.splitext(x)[0]))

        # Load all coordinates
        all_coords = []
        with open(f"{self.coords_dir}/coords.csv", "r") as f:
            reader = csv.reader(f)
            all_coords = [tuple(map(float, row)) for row in reader]

        # Create indices and split if needed
        indices = list(range(len(all_img_files)))

        if split is not None:
            # Shuffle indices deterministically
            rng = random.Random(seed)
            rng.shuffle(indices)

            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            if split == "train":
                indices = indices[:n_train]
            elif split == "val":
                indices = indices[n_train:n_train + n_val]
            elif split == "test":
                indices = indices[n_train + n_val:]
            else:
                raise ValueError(f"split must be 'train', 'val', 'test', or None, got '{split}'")

        # Store only the files/coords for this split
        self.img_files = [all_img_files[i] for i in indices]
        self.coords = [all_coords[i] for i in indices]
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        coords = self.coords[idx]

        # Apply image transform
        if self.img_transform:
            image = self.img_transform(image)

        # Normalize coordinates to [-1, 1] range
        if self.normalize_coords:
            coords = self._normalize_coords(coords)
        else:
            coords = torch.tensor(coords, dtype=torch.float32)

        return image, coords

    def _normalize_coords(self, coords):
        """Normalize lat/lon to [-1, 1] range."""
        lat, lon = coords
        lat = lat / 90.0   # lat: [-90, 90] -> [-1, 1]
        lon = lon / 180.0  # lon: [-180, 180] -> [-1, 1]
        return torch.tensor([lat, lon], dtype=torch.float32)


def get_train_transform(img_size=224):
    """Get training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform(img_size=224):
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_dataloaders(
    root="dataset",
    batch_size=32,
    img_size=224,
    num_workers=4,
    seed=42,
    normalize_coords=True
):
    """
    Create train, val, and test DataLoaders.

    Args:
        root: Path to dataset directory
        batch_size: Batch size for all loaders
        img_size: Image size after resize
        num_workers: Number of workers for data loading
        seed: Random seed for reproducible splits
        normalize_coords: Whether to normalize coordinates

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ImageToGeoDataset(
        root=root,
        split="train",
        seed=seed,
        img_transform=get_train_transform(img_size),
        normalize_coords=normalize_coords
    )

    val_dataset = ImageToGeoDataset(
        root=root,
        split="val",
        seed=seed,
        img_transform=get_val_transform(img_size),
        normalize_coords=normalize_coords
    )

    test_dataset = ImageToGeoDataset(
        root=root,
        split="test",
        seed=seed,
        img_transform=get_val_transform(img_size),
        normalize_coords=normalize_coords
    )

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset and splits
    print("=" * 50)
    print("Testing ImageToGeoDataset")
    print("=" * 50)

    # Test full dataset
    full_dataset = ImageToGeoDataset(
        root="dataset",
        split=None,
        img_transform=get_val_transform(),
        normalize_coords=True
    )
    print(f"\nFull dataset length: {len(full_dataset)}")

    # Test splits
    for split in ["train", "val", "test"]:
        dataset = ImageToGeoDataset(
            root="dataset",
            split=split,
            img_transform=get_val_transform(),
            normalize_coords=True
        )
        print(f"{split.capitalize()} split length: {len(dataset)}")

    # Test a sample
    print("\n" + "=" * 50)
    print("Sample from train split:")
    print("=" * 50)
    train_dataset = ImageToGeoDataset(
        root="dataset",
        split="train",
        img_transform=get_train_transform(),
        normalize_coords=True
    )

    image, coords = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Coords (normalized): {coords.tolist()}")

    # Test DataLoaders
    print("\n" + "=" * 50)
    print("Testing DataLoaders:")
    print("=" * 50)
    train_loader, val_loader, test_loader = get_dataloaders(
        root="dataset",
        batch_size=16,
        num_workers=0  # Use 0 for testing
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a batch
    images, coords = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch coords shape: {coords.shape}")