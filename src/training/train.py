"""
Training pipeline.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.model_loader import save_model
from src.training.validate import validate_model


def train_model(model: Any, train_loader: Any, val_loader: Any, config: Dict[str, Any]) -> Any:
    """
    Train the model using the given data loaders and configuration.

    Must:
      - Set random seeds for reproducibility.
      - Use config-driven hyperparameters (batch_size, epochs,
        learning_rate, optimizer, scheduler).
      - Track training/validation metrics per epoch.
      - Save the best checkpoint via model_loader.save_model().

    Args:
        model: Model instance returned by build_model().
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Full project configuration (training section used).

    Returns:
        Trained model, along with a history of recorded metrics.
    """
    torch.manual_seed(config["data"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tcfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0)
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(tcfg["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_dir = config.get("model", {}).get("checkpoint_dir", "models")
            save_model(model, f"{ckpt_dir}/best.pt", metadata={"epoch": epoch, **val_metrics})

    return model, history


def train_one_epoch(model: Any, train_loader: Any, optimizer: Any, criterion: Any, device: str = "cpu") -> Dict[str, float]:
    """
    Run a single training epoch.

    Args:
        model: Model being trained.
        train_loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.

    Returns:
        Dictionary of epoch training metrics (e.g. {"loss": ..., "accuracy": ...}).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}
