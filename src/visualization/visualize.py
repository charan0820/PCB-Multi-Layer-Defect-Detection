"""
Prediction visualization.

Owner: Person 3 (Inference, Visualization & Application)
"""

from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_prediction(image: Any, prediction: Dict[str, Any]) -> Any:
    """
    Overlay the predicted defect class, confidence, and highlighted
    defect region (heatmap or bounding box) onto the original image.

    Args:
        image: Original (unprocessed) PCB image.
        prediction: Dictionary returned by predict(), optionally
            including localization results (heatmap/bbox).

    Returns:
        Image (or figure) with visual annotations, ready for display
        in the UI or saving to outputs/plots.
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    heatmap = prediction.get("heatmap")
    if heatmap is not None:
        ax.imshow(heatmap, cmap="jet", alpha=0.4)

    bbox = prediction.get("bbox")
    if bbox is not None:
        x, y, w, h = bbox
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"))

    label = prediction.get("class", "UNKNOWN")
    conf = prediction.get("confidence", 0.0)
    ax.set_title(f"{label} ({conf:.1f}%)")
    ax.axis("off")
    return fig


def plot_training_curves(history: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot training/validation loss and accuracy curves.

    Args:
        history: Metrics history returned by train_model().
        save_path: Optional path to save the resulting plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_confusion_matrix(confusion_matrix: Any, class_names: list, save_path: str = None) -> None:
    """
    Plot a confusion matrix for classification results.

    Args:
        confusion_matrix: Precomputed confusion matrix array.
        class_names: List of class label names.
        save_path: Optional path to save the resulting plot.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im)

    if save_path:
        fig.savefig(save_path)
    return fig
