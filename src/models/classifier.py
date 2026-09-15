"""
Model architecture definition.

Owner: Person 2 (ML Model & Training)

Baseline strategy: transfer learning with a lightweight CNN
(MobileNetV3 / EfficientNet-B0 / ResNet18). Start with one baseline
model, then compare alternatives if time permits.
"""

from typing import Any


def build_model(num_classes: int, model_name: str = "efficientnet_b0", pretrained: bool = True) -> Any:
    """
    Build and return the classification model.

    Args:
        num_classes: Number of PCB defect classes (including "no defect"
            if applicable).
        model_name: Backbone architecture identifier, driven by
            config.yaml (model.name).
        pretrained: Whether to initialize with pretrained ImageNet weights.

    Returns:
        Instantiated model ready for training or inference.
    """
    pass


def freeze_backbone(model: Any) -> Any:
    """
    Freeze the backbone layers for initial transfer-learning training,
    leaving only the classification head trainable.

    Args:
        model: Model returned by build_model().

    Returns:
        Model with backbone parameters frozen.
    """
    pass


def unfreeze_backbone(model: Any) -> Any:
    """
    Unfreeze the backbone layers for fine-tuning.

    Args:
        model: Model returned by build_model().

    Returns:
        Model with all parameters trainable.
    """
    pass
