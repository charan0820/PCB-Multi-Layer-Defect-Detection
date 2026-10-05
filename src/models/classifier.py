"""
Model architecture definition.

Owner: Person 2 (ML Model & Training)

Baseline strategy: transfer learning with a lightweight CNN
(MobileNetV3 / EfficientNet-B0 / ResNet18). Start with one baseline
model, then compare alternatives if time permits.
"""

from typing import Any

import torch
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


class FusionModel(nn.Module):
    """
    Dual-branch fusion model: one encoder for the optical image, one for
    the CT scan, feature embeddings concatenated into a shared head.
    """

    def __init__(self, optical_backbone: nn.Module, ct_backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.optical_backbone = optical_backbone
        self.ct_backbone = ct_backbone
        self.head = nn.Linear(feat_dim * 2, num_classes)

    def forward(self, optical, ct):
        f_opt = self.optical_backbone(optical)
        f_ct = self.ct_backbone(ct)
        return self.head(torch.cat([f_opt, f_ct], dim=1))


def build_fusion_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> FusionModel:
    """
    Build a dual-branch optical+CT fusion model, reusing build_model()
    for each branch's backbone with its classification head stripped off.

    Args:
        num_classes: Number of PCB defect classes.
        model_name: Backbone architecture for both branches.
        pretrained: Whether to initialize branches with pretrained weights.

    Returns:
        FusionModel instance.
    """
    optical = build_model(num_classes, model_name, pretrained)
    ct = build_model(num_classes, model_name, pretrained)

    if model_name == "resnet18":
        feat_dim = optical.fc.in_features
        optical.fc = nn.Identity()
        ct.fc = nn.Identity()
    else:
        feat_dim = optical.classifier[-1].in_features
        optical.classifier[-1] = nn.Identity()
        ct.classifier[-1] = nn.Identity()

    if model_name == "resnet18":
        old = ct.conv1
        ct.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=False)
    else:
        old = ct.features[0][0]
        ct.features[0][0] = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                       stride=old.stride, padding=old.padding, bias=False)

    return FusionModel(optical, ct, feat_dim, num_classes)
