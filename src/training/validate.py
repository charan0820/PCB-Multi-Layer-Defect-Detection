"""
Validation loop used during training.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict


def validate_model(model: Any, val_loader: Any, criterion: Any) -> Dict[str, float]:
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
    pass
