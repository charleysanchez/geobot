import os

class ImageDataset():
    def __init__(self, root="dataset"):
        self.root = root

    def get_dataset(self):
        imgs = {}
        contents = os.listdir(self.root)
        for i, path in enumerate(contents):
            if path.endswith("png"):
                imgs[i] = path