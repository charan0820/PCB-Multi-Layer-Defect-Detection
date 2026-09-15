"""
Inference pipeline.

Owner: Person 3 (Inference, Visualization & Application)

Pipeline: load model -> preprocess image -> predict -> visualize -> display result.
"""

from typing import Any, Dict


def predict(image: Any, model: Any) -> Dict[str, Any]:
    """
    Run inference on a single preprocessed PCB image.

    Args:
        image: Preprocessed image (already run through
            preprocessing.preprocess_image()).
        model: Loaded trained model.

    Returns:
        Dictionary containing at minimum:
            {
                "class": <predicted defect class or "PASS">,
                "confidence": <float 0-100>,
                "processing_time_ms": <float>
            }
    """
    pass


def batch_predict(images: Any, model: Any) -> Any:
    """
    Run inference on a batch of preprocessed images.

    Args:
        images: Batch of preprocessed images.
        model: Loaded trained model.

    Returns:
        List of prediction dictionaries, one per image.
    """
    pass
