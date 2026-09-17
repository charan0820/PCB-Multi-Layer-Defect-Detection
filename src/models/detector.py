"""
Defect localization/detection support.

Owner: Person 2 (ML Model & Training)

If the dataset provides bounding boxes/masks, this module can host a
proper detector. Otherwise it hosts CAM/Grad-CAM based explainability
localization used to highlight suspected defect regions without
falsely claiming pixel-level segmentation.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, image: torch.Tensor, class_idx: int):
        self.model.zero_grad()
        output = self.model(image)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def _last_conv(model: nn.Module) -> nn.Module:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1]


def build_localizer(model: Any, method: str = "gradcam") -> Any:
    """
    Build a localization/explainability wrapper around a trained
    classification model.

    Args:
        model: Trained classification model.
        method: Localization method identifier ("gradcam", "cam",
            "attention", or "bbox" if annotations are available).

    Returns:
        A localizer object exposing a method to generate defect
        region heatmaps/boxes for a given image.
    """
    if method != "gradcam":
        raise NotImplementedError(f"Localization method '{method}' not supported")
    return _GradCAM(model, _last_conv(model))


def localize_defect(localizer: Any, image: Any, predicted_class: int) -> Dict[str, Any]:
    """
    Generate a defect localization result for a single image.

    Args:
        localizer: Object returned by build_localizer().
        image: Preprocessed input image.
        predicted_class: Class index predicted by the classifier.

    Returns:
        Dictionary containing e.g. {"heatmap": ..., "bbox": ...}
        depending on the method used.
    """
    return {"heatmap": localizer.generate(image, predicted_class)}
