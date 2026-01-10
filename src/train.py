import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model import ImageToGeoModel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    for images, coords in dataloader:
        # TODO: Move data to device
        images = images.to(device)
        coords = coords.to(device)
        
        # TODO: Zero the gradients
        optimizer.zero_grad()
        
        # TODO: Forward pass
        outputs = model(images)
        
        # TODO: Compute loss
        loss = criterion(outputs, coords)
        
        # TODO: Backward pass
        loss.backward()
        
        # TODO: Update weights
        optimizer.step()
        
        # TODO: Accumulate loss for logging
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    # TODO: What context manager prevents gradient computation?
    # Hint: torch.no_grad()
    
    with torch.no_grad():
    for images, coords in dataloader:
        # TODO: Move data to device
        images = images.to(device)
        coords = coords.to(device)
        
        # TODO: Forward pass
        outputs = model(images)
        
        # TODO: Compute loss
        loss = criterion(outputs, coords)
        
        # TODO: Accumulate loss
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def train(config):
    """Main training function."""
    # TODO: Set device (cuda, mps, or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # TODO: Create dataloaders using your dataset module
    train_loader, val_loader, test_loader = get_dataloaders(
        root=config['dataset_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
        normalize_coords=config['normalize_coords']
    )
    
    # TODO: Initialize model and move to device
    model = ImageToGeoModel()
    model.to(device)
    
    # TODO: Define loss function
    # Hint: For regression, what loss is commonly used?
    criterion = nn.MSELoss()
    
    # TODO: Define optimizer
    # Hint: AdamW(model.parameters(), lr=???, weight_decay=???)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # TODO: Define learning rate scheduler (optional but recommended)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # TODO: Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # TODO: Validate
        val_loss = validate(model, val_loader, criterion, device)

        # TODO: Step the scheduler
        scheduler.step()
        
        # TODO: Print progress
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # TODO: Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Hint: torch.save({...}, "checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, "checkpoint.pt")


if __name__ == "__main__":
    config = {
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'img_size': 224,
    }
    
    train(config)