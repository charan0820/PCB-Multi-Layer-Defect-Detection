"""
Inference pipeline.

Owner: Person 3 (Inference, Visualization & Application)

Pipeline: load model -> preprocess image -> predict -> visualize -> display result.
"""

from typing import Any, Dict
import time

import torch
import torch.nn.functional as F


def _to_tensor(image: Any) -> torch.Tensor:
    x = torch.as_tensor(image).float()
    if x.dim() == 3:
        x = x.permute(2, 0, 1)
    return x.unsqueeze(0)


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
    model.eval()
    x = _to_tensor(image)
    t0 = time.time()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        conf, idx = probs.max(dim=1)
    elapsed_ms = (time.time() - t0) * 1000
    return {
        "class": int(idx.item()),
        "confidence": float(conf.item() * 100),
        "processing_time_ms": elapsed_ms,
    }


def batch_predict(images: Any, model: Any) -> Any:
    """
    Run inference on a batch of preprocessed images.

    Args:
        images: Batch of preprocessed images.
        model: Loaded trained model.

    Returns:
        List of prediction dictionaries, one per image.
    """
    return [predict(img, model) for img in images]
