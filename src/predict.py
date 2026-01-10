import torch
import argparse
from PIL import Image

from src.model import ImageToGeoModel
from src.dataset import get_val_transform


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_name = checkpoint.get('backbone_name', 'mobilenetv3_large_100')
    model = ImageToGeoModel(backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, device):
    """Predict coordinates for a single image."""
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transform()
    image = transform(image)
    
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(image)
    
    # Denormalize predictions
    lat = predictions[0, 0].item() * 90
    lon = predictions[0, 1].item() * 180
    
    return lat, lon


def main():
    parser = argparse.ArgumentParser(description="Predict coordinates from image")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to checkpoint")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Run prediction
    lat, lon = predict_image(model, args.image_path, device)
    
    # Print results
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"Predicted location: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")


if __name__ == "__main__":
    main()