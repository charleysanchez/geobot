from tqdm import tqdm
import torch
import torch.nn as nn
import math
import yaml
import argparse
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model import ImageToGeoModel

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for images, coords in pbar:
        images = images.to(device)
        coords = coords.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coords)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """Validate the model with progress bar."""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for images, coords in pbar:
            images = images.to(device)
            coords = coords.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, coords)
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


def train(config, use_wandb=False):
    """Main training function."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="geobot",
            config=config,
            name=f"{config['backbone_name']}_{'haversine' if config.get('use_haversine_loss') else 'mse'}"
        )
        print("✓ Weights & Biases logging enabled")
    
    # Print training config
    print("\n" + "=" * 50)
    print("GEOBOT TRAINING")
    print("=" * 50)
    print(f"Device:          {device}")
    print(f"Backbone:        {config['backbone_name']}")
    print(f"Loss:            {'Haversine (km)' if config.get('use_haversine_loss') else 'MSE'}")
    print(f"Batch size:      {config['batch_size']}")
    print(f"Learning rate:   {config['learning_rate']}")
    print(f"Epochs:          {config['num_epochs']}")
    
    # Freeze/unfreeze settings
    freeze_backbone = config.get('freeze_backbone', False)
    unfreeze_at_epoch = config.get('unfreeze_at_epoch', 0)
    unfreeze_lr = config.get('unfreeze_lr', config['learning_rate'] / 10)
    
    if freeze_backbone:
        print(f"Freeze backbone: Yes (unfreeze at epoch {unfreeze_at_epoch})")
        print(f"Unfreeze LR:     {unfreeze_lr}")
    print("=" * 50 + "\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root=config['dataset_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
        normalize_coords=config['normalize_coords']
    )
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    
    # Initialize model
    model = ImageToGeoModel(config['backbone_name'])
    model.to(device)
    
    # Freeze backbone if configured
    if freeze_backbone:
        model.freeze_backbone()
    
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Loss, optimizer, scheduler
    if config.get('use_haversine_loss', False):
        criterion = HaversineLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    best_val_loss = float('inf')
    
    print("\nStarting training...\n")
    
    for epoch in range(config['num_epochs']):
        # Check if we should unfreeze backbone
        if freeze_backbone and epoch == unfreeze_at_epoch:
            model.unfreeze_backbone()
            print(f"Trainable params: {model.get_trainable_params():,}")
            # Reset optimizer with lower learning rate for fine-tuning
            optimizer = AdamW(model.parameters(), lr=unfreeze_lr, weight_decay=config['weight_decay'])
            scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'] - epoch)
            print(f"Optimizer reset with LR: {unfreeze_lr}")
        
        # Train and validate
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config['num_epochs'])
        val_loss = validate(model, val_loader, criterion, device, epoch, config['num_epochs'])
        scheduler.step()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })
        
        # Print epoch summary
        loss_unit = "km" if config.get('use_haversine_loss') else ""
        improved = "NEW BEST" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} │ Train: {train_loss:8.4f}{loss_unit} │ Val: {val_loss:8.4f}{loss_unit} │ LR: {current_lr:.2e} {improved}")
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = config.get('checkpoint_path', 'checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'backbone_name': config['backbone_name'],
                'config': config,
            }, checkpoint_path)
    
    print("\n" + "=" * 50)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("=" * 50)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoBot model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {args.config}")
    
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")
    
    train(config, use_wandb=args.wandb)