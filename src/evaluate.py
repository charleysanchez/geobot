import torch
import math
import argparse

from src.model import ImageToGeoModel
from src.dataset import get_dataloaders


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point (in degrees)
        lat2, lon2: Second point (in degrees)
    
    Returns:
        Distance in kilometers
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return 6371 * c


def denormalize_coords(normalized_coords):
    """
    Convert normalized coords [-1, 1] back to degrees.
    
    Args:
        normalized_coords: Tensor of shape (N, 2) with [lat, lon] in [-1, 1]
    
    Returns:
        Tensor with [lat, lon] in degrees
    """
    lat = normalized_coords[:, 0] * 90
    lon = normalized_coords[:, 1] * 180
    return torch.stack([lat, lon], dim=1)


def evaluate(model, dataloader, device):
    """
    Evaluate model and compute metrics.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_distances = []
    
    with torch.no_grad():
        for images, coords in dataloader:
            images = images.to(device)
            coords = coords.to(device)
            
            predictions = model(images)
            
            # Denormalize both
            predictions = denormalize_coords(predictions)
            coords = denormalize_coords(coords)
            
            # Calculate haversine distance for each sample
            for pred, true in zip(predictions, coords):
                dist = haversine_distance(
                    pred[0].item(), pred[1].item(),
                    true[0].item(), true[1].item()
                )
                all_distances.append(dist)
    
    # Convert to tensor for easy calculations
    distances = torch.tensor(all_distances)
    
    return {
        'mean_distance_km': distances.mean().item(),
        'median_distance_km': distances.median().item(),
        'accuracy_1km': (distances < 1).float().mean().item() * 100,
        'accuracy_25km': (distances < 25).float().mean().item() * 100,
        'accuracy_200km': (distances < 200).float().mean().item() * 100,
        'accuracy_750km': (distances < 750).float().mean().item() * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoBot model")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model (use same backbone as training)
    backbone_name = checkpoint.get('backbone_name', 'mobilenetv3_large_100')
    model = ImageToGeoModel(backbone_name=backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model: {backbone_name}")
    
    # Create test dataloader
    _, _, test_loader = get_dataloaders(
        root=args.dataset,
        batch_size=args.batch_size,
        img_size=224,
        num_workers=4,
        seed=42,
        normalize_coords=True
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Run evaluation
    print("\nEvaluating on test set...")
    metrics = evaluate(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean distance:    {metrics['mean_distance_km']:,.2f} km")
    print(f"Median distance:  {metrics['median_distance_km']:,.2f} km")
    print("-" * 50)
    print(f"Accuracy @ 1km:   {metrics['accuracy_1km']:.2f}%")
    print(f"Accuracy @ 25km:  {metrics['accuracy_25km']:.2f}%")
    print(f"Accuracy @ 200km: {metrics['accuracy_200km']:.2f}%")
    print(f"Accuracy @ 750km: {metrics['accuracy_750km']:.2f}%")
    print("=" * 50)