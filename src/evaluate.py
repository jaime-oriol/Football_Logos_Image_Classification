"""
Model evaluation and metrics visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate model on test set and display metrics.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to evaluate on

    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    print(f'Test Accuracy: {accuracy:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, predictions, class_names, figsize=(12, 10)):
    """
    Plot confusion matrix heatmap.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.

    Args:
        history: Dictionary with training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
