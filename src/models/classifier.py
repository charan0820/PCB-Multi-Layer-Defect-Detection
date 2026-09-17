"""
Model architecture definition.

Owner: Person 2 (ML Model & Training)

Baseline strategy: transfer learning with a lightweight CNN
(MobileNetV3 / EfficientNet-B0 / ResNet18). Start with one baseline
model, then compare alternatives if time permits.
"""

from typing import Any

import torch.nn as nn
import torchvision.models as tvm

_ARCHS = {
    "resnet18": ("fc",),
    "mobilenet_v3": ("classifier", -1),
    "efficientnet_b0": ("classifier", -1),
}
_BUILDERS = {
    "resnet18": tvm.resnet18,
    "mobilenet_v3": tvm.mobilenet_v3_small,
    "efficientnet_b0": tvm.efficientnet_b0,
}


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
    builder = _BUILDERS.get(model_name, _BUILDERS["efficientnet_b0"])
    model = builder(weights="DEFAULT" if pretrained else None)
    if model_name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


def freeze_backbone(model: Any) -> Any:
    """
    Freeze the backbone layers for initial transfer-learning training,
    leaving only the classification head trainable.

    Args:
        model: Model returned by build_model().

    Returns:
        Model with backbone parameters frozen.
    """
    for name, param in model.named_parameters():
        if "fc" not in name and "classifier" not in name:
            param.requires_grad = False
    return model


def unfreeze_backbone(model: Any) -> Any:
    """
    Unfreeze the backbone layers for fine-tuning.

    Args:
        model: Model returned by build_model().

    Returns:
        Model with all parameters trainable.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model
