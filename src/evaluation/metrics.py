"""
Model evaluation metrics.

Owner: Person 3 (Inference, Visualization & Application)
"""

from typing import Any, Dict


def evaluate_model(model: Any, test_loader: Any) -> Dict[str, Any]:
    """
    Evaluate a trained model on the held-out test set.

    Computes classification metrics (accuracy, precision, recall,
    F1, confusion matrix), and, if localization annotations are
    available, IoU and localization precision/recall. Also measures
    system-level metrics: inference time, model size, image
    processing time.

    Args:
        model: Trained model to evaluate.
        test_loader: Test DataLoader.

    Returns:
        Dictionary of computed metrics.
    """
    pass


def compute_iou(pred_box: Any, gt_box: Any) -> float:
    """
    Compute Intersection-over-Union between a predicted and
    ground-truth bounding box, when localization annotations exist.

    Args:
        pred_box: Predicted bounding box (x, y, w, h) or (x1, y1, x2, y2).
        gt_box: Ground-truth bounding box in the same format.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    pass
