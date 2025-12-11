# Football Logos Image Classification

Classification of football team logos into their respective European leagues using Deep Learning.

## Project Overview

**Task:** Multi-class image classification (26 leagues)
**Dataset:** 605 unique team logos from 26 European football leagues
**Models:** Custom CNN baseline and ResNet18 transfer learning

## Dataset Structure

```
data/
├── Austria - Bundesliga/          (16 teams)
├── Belgium - Jupiler Pro League/  (28 teams)
├── England - Premier League/      (31 teams)
├── Spain - LaLiga/                (26 teams)
└── ... (26 leagues total)
```

**Total:** 605 logos, average 23.3 teams per league

## Project Structure

```
Football_Logos_Image_Classification/
├── data/                   # Dataset (26 league folders with logos)
├── src/                    # Python modules
│   ├── dataset.py         # DataLoaders and transformations
│   ├── models.py          # CNN architectures
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation and metrics
│   └── utils.py           # Visualization and prediction
├── notebooks/
│   └── main.ipynb         # Main notebook
├── models/                # Saved model weights
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Main Notebook

```bash
cd notebooks
jupyter notebook main.ipynb
```

The notebook includes:
1. Dataset loading and exploration
2. Training Custom CNN baseline
3. Training ResNet18 with transfer learning
4. Evaluation and comparison
5. Confusion matrix visualization
6. Prediction on custom images

### Quick Example

```python
from src.dataset import get_dataloaders
from src.models import get_resnet18
from src.train import train_model

# Load data
train_loader, val_loader, test_loader, classes = get_dataloaders('data/')

# Create model
model = get_resnet18(num_classes=26, pretrained=True)

# Train
history = train_model(model, train_loader, val_loader, epochs=15)
```

## Models

### Model 1: Custom CNN
- 3 convolutional blocks
- Baseline architecture
- Expected accuracy: 70-75%

### Model 2: ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned for 26 leagues
- Expected accuracy: 85-90%

## Data Augmentation

**Training:**
- Resize to 224x224
- Random horizontal flip
- Color jitter (brightness, contrast)
- ImageNet normalization

**Testing:**
- Resize to 224x224
- ImageNet normalization

## Evaluation Metrics

- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Classification report

## Results

Results will be available after running the notebook. Expected performance:
- Custom CNN: ~70-75% test accuracy
- ResNet18: ~85-90% test accuracy

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, recommended)

## Author

Jaime Oriol

## License

This project is for educational purposes as part of a Machine Learning course assignment.
