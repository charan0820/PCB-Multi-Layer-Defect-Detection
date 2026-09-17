"""
Model evaluation metrics.

Owner: Person 3 (Inference, Visualization & Application)
"""

from typing import Any, Dict
import time

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
    model.eval()
    all_preds, all_labels = [], []
    img_times = []
    start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.time()
            outputs = model(images)
            img_times.append((time.time() - t0) / images.size(0))
            all_preds.extend(outputs.argmax(1).tolist())
            all_labels.extend(labels.tolist())
    total_time = time.time() - start

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "inference_time_s": total_time,
        "avg_image_time_s": sum(img_times) / len(img_times) if img_times else 0.0,
        "model_size_params": n_params,
    }


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
