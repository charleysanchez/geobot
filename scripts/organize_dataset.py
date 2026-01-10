import os

def create_image_dir(root="dataset"):
    os.mkdir(f"{root}/images")
    os.mkdir(f"{root}/coords")

    for path in os.listdir(root):
        if path.endswith(".png"):
            print(path)


if __name__ == "__main__":
    create_image_dir()