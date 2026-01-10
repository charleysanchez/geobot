# 🌍 GeoBot

A deep learning model that predicts geographic coordinates from Street View images. Given a street-level photograph, GeoBot estimates the latitude and longitude of where the image was taken.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

GeoBot uses transfer learning with modern CNN architectures to solve the challenging problem of visual geolocation. The model is trained on ~10,000 Google Street View images from around the world and learns to recognize visual cues (architecture, vegetation, road signs, etc.) that indicate geographic location.

### Key Features
- **Transfer Learning**: Leverages pretrained backbones (MobileNetV3, EfficientNet, DenseNet, etc.)
- **Flexible Architecture**: Easily swap between different `timm` model backbones
- **Efficient Training**: Data augmentation, learning rate scheduling, and checkpoint saving
- **Robust Evaluation**: Haversine distance metrics and accuracy at multiple thresholds

## Project Structure

```
geobot/
├── src/
│   ├── dataset.py    # Data loading with train/val/test splits
│   ├── model.py      # CNN architecture with configurable backbone
│   ├── train.py      # Training loop with checkpointing
│   ├── evaluate.py   # Evaluation metrics (haversine distance)
│   └── predict.py    # Single-image inference
├── dataset/
│   ├── images/       # Street View images
│   └── coords/       # Coordinate labels (CSV)
└── scripts/
    └── organize_dataset.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geobot.git
cd geobot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision timm pillow tqdm
```

## Dataset

Download the Google Street View dataset from Kaggle:

```bash
curl -L -o ~/Downloads/google-street-view.zip \
  https://www.kaggle.com/api/v1/datasets/download/paulchambaz/google-street-view

unzip ~/Downloads/google-street-view.zip -d dataset/
python scripts/organize_dataset.py
```

**Dataset Stats:**
- 10,000 images (640x640 PNG)
- Global coverage
- Train/Val/Test split: 80%/10%/10%

## Usage

### Training

```bash
python -m src.train
```

Training uses:
- **Backbone**: MobileNetV3 (configurable)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing
- **Loss**: MSE on normalized coordinates

### Evaluation

```bash
python -m src.evaluate
```

Outputs distance metrics in kilometers:
- Mean/Median haversine distance
- Accuracy at 1km, 25km, 200km, 750km thresholds

### Inference

```bash
# Predict location from a single image
python -m src.predict path/to/streetview.jpg

# Output: Predicted location: 48.8566°N, 2.3522°E
```

## Model Architecture

```
Input Image (3 x 224 x 224)
         ↓
┌─────────────────────┐
│  Pretrained Backbone │  (MobileNetV3, EfficientNet, etc.)
│    (timm library)    │
└─────────────────────┘
         ↓
    Feature Vector
         ↓
┌─────────────────────┐
│  Dropout (0.2)      │
│  Linear → 2         │  (lat, lon)
└─────────────────────┘
         ↓
  Normalized Coords [-1, 1]
```

## Configuration

Edit the config in `src/train.py`:

```python
config = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'img_size': 224,
    'backbone_name': 'mobilenetv3_large_100',  # Try: efficientnet_b0, resnet18
}
```

## Future Improvements

- [ ] Implement Haversine loss function (distance-aware training)
- [ ] Add region classification head (coarse-to-fine prediction)
- [ ] Experiment with Vision Transformers (ViT)
- [ ] Build interactive web demo with Gradio
- [ ] Fine-tune on specific regions for higher accuracy

## Acknowledgments

- Dataset: [Google Street View Dataset](https://www.kaggle.com/datasets/paulchambaz/google-street-view) by Paul Chambaz
- Pretrained models: [timm](https://github.com/huggingface/pytorch-image-models) library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
