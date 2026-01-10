# 🌍 GeoBot

A deep learning model that predicts geographic coordinates from Street View images. Given a street-level photograph, GeoBot estimates the latitude and longitude of where the image was taken.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

GeoBot uses transfer learning with modern CNN architectures to solve the challenging problem of visual geolocation. The model is trained on ~10,000 Google Street View images from around the world and learns to recognize visual cues (architecture, vegetation, road signs, etc.) that indicate geographic location.

### Key Features
- **Transfer Learning**: Leverages pretrained backbones (MobileNetV3, EfficientNet, ResNet, etc.)
- **YAML Configs**: Easy experiment management with config files
- **Weights & Biases**: Optional experiment tracking with `--wandb` flag
- **Haversine Loss**: Train with actual geographic distance in km
- **Gradio Demo**: Interactive web app with map visualization

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/geobot.git
cd geobot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset
curl -L -o ~/Downloads/google-street-view.zip \
  https://www.kaggle.com/api/v1/datasets/download/paulchambaz/google-street-view
unzip ~/Downloads/google-street-view.zip -d dataset/
python scripts/organize_dataset.py

# Train
python -m src.train --config configs/default.yaml

# Launch demo
python app.py
```

## Project Structure

```
geobot/
├── app.py                 # Gradio web demo
├── configs/               # YAML config files
│   ├── default.yaml
│   ├── haversine_loss.yaml
│   ├── efficientnet.yaml
│   └── resnet18.yaml
├── src/
│   ├── dataset.py         # Data loading with train/val/test splits
│   ├── model.py           # CNN architecture with configurable backbone
│   ├── train.py           # Training loop with wandb support
│   ├── evaluate.py        # Haversine distance metrics
│   └── predict.py         # Single-image inference
└── dataset/
    ├── images/            # Street View images
    └── coords/            # Coordinate labels (CSV)
```

## Training

```bash
# Default config
python -m src.train

# Custom config
python -m src.train --config configs/efficientnet.yaml

# With Weights & Biases logging
python -m src.train --config configs/default.yaml --wandb
```

### Configuration Options

See `configs/default.yaml`:
```yaml
num_epochs: 50
batch_size: 32
learning_rate: 1.0e-4
backbone_name: mobilenetv3_large_100
use_haversine_loss: true  # Train with distance loss (km)
```

## Evaluation

```bash
python -m src.evaluate --checkpoint checkpoint.pt
```

Outputs:
- Mean/Median haversine distance (km)
- Accuracy at 1km, 25km, 200km, 750km thresholds

## Inference

```bash
# Single image prediction
python -m src.predict path/to/streetview.jpg
```

## Web Demo

```bash
python app.py
# Open http://localhost:7860
```

Features:
- Upload any Street View image
- See predicted coordinates on interactive map
- Direct link to Google Maps

## Model Architecture

```
Input Image (3 x 224 x 224)
         ↓
┌─────────────────────┐
│  Pretrained Backbone │  (MobileNetV3, EfficientNet, etc.)
│    (timm library)    │
└─────────────────────┘
         ↓
     Dropout (0.2)
         ↓
     Linear → 2 (lat, lon)
         ↓
  Normalized Coords [-1, 1]
```

## Acknowledgments

- Dataset: [Google Street View Dataset](https://www.kaggle.com/datasets/paulchambaz/google-street-view) by Paul Chambaz
- Pretrained models: [timm](https://github.com/huggingface/pytorch-image-models) library

## License

MIT License - see [LICENSE](LICENSE) for details.
