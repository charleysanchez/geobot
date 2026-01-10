from torch.utils.data import Dataset
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
        ])

        self.coords = []
        with open(f"{self.coords_dir}/coords.csv", "r") as f:
            reader = csv.reader(f)
            self.coords = list(reader)

        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        