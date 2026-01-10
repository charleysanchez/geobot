import timm
import torch.nn as nn

class ImageToGeoModel(nn.Module):
    def __init__(self, backbone_name="densenet121", img_size=224):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
