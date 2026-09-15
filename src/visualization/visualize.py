"""
Prediction visualization.

Owner: Person 3 (Inference, Visualization & Application)
"""

from typing import Any, Dict


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
    pass


def plot_training_curves(history: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot training/validation loss and accuracy curves.

    Args:
        history: Metrics history returned by train_model().
        save_path: Optional path to save the resulting plot.
    """
    pass


def plot_confusion_matrix(confusion_matrix: Any, class_names: list, save_path: str = None) -> None:
    """
    Plot a confusion matrix for classification results.

    Args:
        confusion_matrix: Precomputed confusion matrix array.
        class_names: List of class label names.
        save_path: Optional path to save the resulting plot.
    """
    pass
