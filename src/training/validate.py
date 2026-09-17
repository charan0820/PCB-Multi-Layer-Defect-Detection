"""
Validation loop used during training.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict

import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def validate_model(model: Any, val_loader: Any, criterion: Any, device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate the model on the validation set for one epoch.

    Args:
        model: Model being validated.
        val_loader: Validation DataLoader.
        criterion: Loss function.

    Returns:
        Dictionary of validation metrics (e.g. loss, accuracy,
        recall — recall is prioritized per project requirements
        since missing a real defect is worse than a false alarm).
    """
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": sum(p == l for p, l in zip(all_preds, all_labels)) / total,
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
