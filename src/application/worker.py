"""Private inference worker, run via python -m (including on Windows)."""
import json
from pathlib import Path
import sys
import traceback


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    try:
        from PIL import Image
        import torch
        from src.models.model_loader import load_model
        from src.inference.predict import predict
        torch.set_num_threads(1)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        model = load_model(request["checkpoint"])
        names = model.names
        names = list(names.values()) if isinstance(names, dict) else list(names)
        output = {"names": names, "model_name": Path(request["checkpoint"]).name}
        if request.get("image"):
            with Image.open(request["image"]) as source:
                output["prediction"] = predict(
                    source.convert("RGB"), model, confidence_threshold=0.01,
                    device=request["device"], imgsz=request["imgsz"], iou=request["iou"],
                )
            output["prediction"]["model_name"] = output["model_name"]
        result_path.write_text(json.dumps(output), encoding="utf-8")
    except Exception as exc:
        traceback.print_exc()
        result_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8")


if __name__ == "__main__":
    main()
