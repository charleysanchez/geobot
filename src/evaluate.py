import torch
import math

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
    # Device setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load("checkpoint.pt", map_location=device)
    
    # Initialize model (use same backbone as training)
    backbone_name = checkpoint.get('backbone_name', 'mobilenetv3_large_100')
    model = ImageToGeoModel(backbone_name=backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataloader
    _, _, test_loader = get_dataloaders(
        root='dataset',
        batch_size=32,
        img_size=224,
        num_workers=4,
        seed=42,
        normalize_coords=True
    )
    
    # Run evaluation
    print("\nEvaluating on test set...")
    metrics = evaluate(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")