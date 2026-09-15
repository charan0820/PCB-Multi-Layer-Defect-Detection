"""
Model checkpoint saving/loading.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict


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
    pass


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
    pass
