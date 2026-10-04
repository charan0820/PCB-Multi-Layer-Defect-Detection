"""
Configuration loading utilities.

Centralizes access to config.yaml so no parameters are hard-coded
throughout the project.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the project configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration parameters.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_param(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example:
        get_param(config, "training.batch_size")

    Args:
        config: Loaded configuration dictionary.
        key_path: Dot-separated path to the desired parameter.
        default: Value returned if the key path is not found.

    Returns:
        The requested configuration value, or default if missing.
    """
    node = config
    for key in key_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
