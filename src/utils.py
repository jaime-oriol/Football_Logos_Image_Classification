"""
Utility functions for visualization and prediction.
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def predict_image(image_path, model, class_names, device=None, top_k=3):
    """
    Predict class for a single image.

    Args:
        image_path: Path to image file
        model: Trained PyTorch model
        class_names: List of class names
        device: Device to run prediction on
        top_k: Number of top predictions to return

    Returns:
        List of tuples (class_name, probability)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append((class_names[idx], prob.item() * 100))

    return results


def visualize_prediction(image_path, predictions):
    """
    Visualize image with top predictions.

    Args:
        image_path: Path to image file
        predictions: List of tuples (class_name, probability)
    """
    image = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')

    classes = [pred[0] for pred in predictions]
    probs = [pred[1] for pred in predictions]

    ax2.barh(classes, probs)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    ax2.set_xlim([0, 100])

    for i, (cls, prob) in enumerate(predictions):
        ax2.text(prob + 2, i, f'{prob:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


def predict_from_dataset(dataset, model, class_names, idx, device=None, top_k=3):
    """
    Predict class for an image from dataset.

    Args:
        dataset: PyTorch dataset
        model: Trained PyTorch model
        class_names: List of class names
        idx: Index of image in dataset
        device: Device to run prediction on
        top_k: Number of top predictions to return

    Returns:
        Tuple of (predictions, true_label, image_tensor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image, true_label = dataset[idx]
    image_batch = image.unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append((class_names[idx], prob.item() * 100))

    return predictions, class_names[true_label], image


def visualize_prediction_from_dataset(image_tensor, predictions, true_label):
    """
    Visualize prediction for dataset image.

    Args:
        image_tensor: Image tensor from dataset
        predictions: List of tuples (class_name, probability)
        true_label: True class name
    """
    image = image_tensor.permute(1, 2, 0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'True Label: {true_label}', fontsize=12, fontweight='bold')

    classes = [pred[0] for pred in predictions]
    probs = [pred[1] for pred in predictions]

    colors = ['green' if cls == true_label else 'gray' for cls in classes]
    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    ax2.set_xlim([0, 100])

    for i, (cls, prob) in enumerate(predictions):
        ax2.text(prob + 2, i, f'{prob:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


def visualize_dataset_samples(dataset, class_names, n_samples=16):
    """
    Display random samples from dataset.

    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        n_samples: Number of samples to display
    """
    indices = torch.randperm(len(dataset))[:n_samples]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image = image * std + mean
            image = torch.clamp(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(class_names[label], fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
