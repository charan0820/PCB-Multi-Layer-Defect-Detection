"""
Defect localization/detection support.

Owner: Person 2 (ML Model & Training)

If the dataset provides bounding boxes/masks, this module can host a
proper detector. Otherwise it hosts CAM/Grad-CAM based explainability
localization used to highlight suspected defect regions without
falsely claiming pixel-level segmentation.
"""

from typing import Any, Dict


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
    pass


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
    pass
