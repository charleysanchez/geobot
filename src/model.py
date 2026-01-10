import timm
import torch
import torch.nn as nn

class ImageToGeoModel(nn.Module):
    def __init__(self, backbone_name="densenet121", img_size=224):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # Get actual feature dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            num_features = self.backbone(dummy).shape[1]
        
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
