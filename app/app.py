"""
Simple UI application for the PCB Multi-Layer Defect Detection prototype.

Owner: Person 3 (Inference, Visualization & Application)

IMPORTANT: This file must NOT contain model-training code. It only
wires together: load model -> preprocess image -> predict -> visualize
-> display result.

Flow:
    1. Upload PCB image
    2. Display original image
    3. Run model
    4. Display predicted defect class
    5. Display confidence
    6. Display highlighted defect region
    7. Display PASS/DEFECTIVE status
    8. Display inference time
"""

from typing import Any

# from src.utils.config import load_config
# from src.models.model_loader import load_model
# from src.data.preprocessing import preprocess_image
# from src.inference.predict import predict
# from src.visualization.visualize import visualize_prediction


def load_app_model(config: dict) -> Any:
    """
    Load the trained model for use by the UI, based on config
    (checkpoint path, model name, num_classes).

    Args:
        config: Loaded project configuration.

    Returns:
        Loaded model instance.
    """
    pass


def run_inference_pipeline(uploaded_image: Any, model: Any, config: dict) -> dict:
    """
    Run the full UI-facing pipeline for a single uploaded image:
    preprocess -> predict -> visualize.

    Args:
        uploaded_image: Raw image uploaded by the user via the UI.
        model: Loaded model instance.
        config: Loaded project configuration.

    Returns:
        Dictionary with prediction results and the visualized image,
        ready to be rendered in the UI.
    """
    pass


def main() -> None:
    """
    Entry point for the UI application (e.g. Streamlit app).

    Should not contain any training logic. Only orchestrates the
    upload -> preprocess -> predict -> visualize -> display flow.
    """
    pass


if __name__ == "__main__":
    main()
