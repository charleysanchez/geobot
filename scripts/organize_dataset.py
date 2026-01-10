import os

def create_image_dir(root="dataset"):
    img_path = os.path.join(root, "images")
    coords_path = os.path.join(root, "coords")
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    
    if not os.path.exists(coords_path):
        os.mkdir(coords_path)

    for file in os.listdir(root):
        if file.endswith(".png"):
            os.rename(os.path.join(root, file), os.path.join(img_path, file))
        elif file.endswith(".csv"):
            os.rename(os.path.join(root, file), os.path.join(coords_path, file))


if __name__ == "__main__":
    create_image_dir()