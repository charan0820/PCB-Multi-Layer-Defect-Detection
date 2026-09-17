"""
Model checkpoint saving/loading.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict
import json
from pathlib import Path

import torch

from src.models.classifier import build_model


def save_model(model: Any, path: str, metadata: Dict[str, Any] = None) -> None:
    """
    Save a trained model checkpoint along with metadata (dataset
    version, hyperparameters, epochs, metrics) for reproducibility.

    Args:
        model: Trained model instance.
        path: Destination file path (config-driven checkpoint_dir).
        metadata: Additional reproducibility metadata to store
            alongside the weights.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if metadata:
        with open(f"{path}.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_model(path: str, model_name: str, num_classes: int) -> Any:
    """
    Load a model checkpoint from disk.

    Args:
        path: Path to the saved checkpoint.
        model_name: Backbone architecture identifier used to
            reconstruct the model before loading weights.
        num_classes: Number of output classes.

    Returns:
        Model instance with loaded weights, ready for inference.
    """
    model = build_model(num_classes, model_name, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
