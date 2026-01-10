from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import csv
import os

class ImageToGeoDataset(Dataset):
    def __init__(self, root="dataset", img_transform=None):
        self.img_dir = os.path.join(root, "images")
        self.coords_dir = os.path.join(root, "coords")

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ], key=lambda x: int(os.path.splitext(x)[0]))

        self.coords = []
        with open(f"{self.coords_dir}/coords.csv", "r") as f:
            reader = csv.reader(f)
            self.coords = [tuple(map(float, row)) for row in reader]

        print(self.coords)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        coords = self.coords[idx]

        return (
            self.img_transform(image) if self.img_transform else image,
            torch.tensor(coords, dtype=torch.float32)
        )
    
if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),   # <-- converts PIL → torch.Tensor (C,H,W)
        ])
    dataset = ImageToGeoDataset(
        root="dataset",
        img_transform=transform
    )

    print("Dataset length:", len(dataset))
    image, coords = dataset[0]

    print(type(image), image.shape)
    print(type(coords), coords.shape, coords, coords.dtype)

    for i in range(5):
        img, c = dataset[i]
        print(i, dataset.img_files[i], c.tolist())
        