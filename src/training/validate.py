"""
Validation loop used during training.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict

import torch


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
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}
