from tqdm import tqdm
import torch
import torch.nn as nn
import math
import yaml
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model import ImageToGeoModel


class HaversineLoss(nn.Module):
    """
    Loss function that computes the mean haversine distance in km.
    Expects normalized coordinates in [-1, 1] range.
    """
    def __init__(self, earth_radius=6371.0):
        super().__init__()
        self.earth_radius = earth_radius
    
    def forward(self, pred, target):
        # Denormalize: [-1, 1] -> degrees
        pred_lat = pred[:, 0] * 90
        pred_lon = pred[:, 1] * 180
        target_lat = target[:, 0] * 90
        target_lon = target[:, 1] * 180
        
        # Convert to radians
        pred_lat = pred_lat * math.pi / 180
        pred_lon = pred_lon * math.pi / 180
        target_lat = target_lat * math.pi / 180
        target_lon = target_lon * math.pi / 180
        
        # Haversine formula
        dlat = target_lat - pred_lat
        dlon = target_lon - pred_lon
        
        a = torch.sin(dlat / 2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon / 2)**2
        c = 2 * torch.asin(torch.sqrt(a.clamp(min=1e-8, max=1.0)))  # clamp for numerical stability
        
        distance_km = self.earth_radius * c
        
        return distance_km.mean()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, coords in tqdm(dataloader):
        images = images.to(device)
        coords = coords.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coords)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, coords in dataloader:
            images = images.to(device)
            coords = coords.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, coords)
            total_loss += loss.item()
        
    return total_loss / len(dataloader)


def train(config):
    """Main training function."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root=config['dataset_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
        normalize_coords=config['normalize_coords']
    )

    print(f"Finetuning Pretrained Model: {config['backbone_name']}")
    
    # Initialize model
    model = ImageToGeoModel(config['backbone_name'])
    model.to(device)
    
    # Loss, optimizer, scheduler
    if config.get('use_haversine_loss', False):
        criterion = HaversineLoss()
        print("Using Haversine loss (distance in km)")
    else:
        criterion = nn.MSELoss()
        print("Using MSE loss")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(config['num_epochs'])):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'backbone_name': config['backbone_name'],
            }, "checkpoint.pt")
            print(f"  ✓ Saved new best checkpoint (val_loss: {val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoBot model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {args.config}")
    train(config)