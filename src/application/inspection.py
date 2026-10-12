"""Bounded image decoding and isolated inference for the interactive app."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000
PCB_CLASSES = frozenset(("mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"))


def decode_image(data: bytes) -> Image.Image:
    """Validate size/format, apply EXIF orientation and return detached RGB pixels."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 15 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image exceeds 20 megapixels. Crop or resize it before inspection.")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")


def resolve_checkpoint(value: str) -> Path:
    """Use exactly the selected checkpoint; never fall back to generic weights."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"Trained checkpoint not found: {path}. Copy your trained best.pt here.")
    return path


def inspection_key(data: bytes, checkpoint: Path, device: str, imgsz: int, iou: float) -> str:
    stat = checkpoint.stat()
    identity = f"{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}:{device}:{imgsz}:{iou}"
    return hashlib.sha256(data + identity.encode()).hexdigest()


def run_worker(checkpoint: Path, image=None, *, device="cpu", imgsz=640, iou=0.45, timeout=120):
    """A native PyTorch/CUDA failure kills the worker, not the Streamlit server.

    Each scan loads its own model. Sliders and downloads use plain cached data.
    Worker diagnostics are retained even for access violations and timeouts.
    """
    logs = ROOT / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pcb-inspect-") as directory:
        directory = Path(directory)
        request = {"checkpoint": str(checkpoint), "device": device, "imgsz": imgsz, "iou": iou}
        if image is not None:
            image.save(directory / "input.png")
            request["image"] = str(directory / "input.png")
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (logs / "inference.log").open("a", encoding="utf-8") as log:
            log.write(f"\nCheckpoint={checkpoint}; device={device}; image={image is not None}\n")
            log.flush()
            try:
                process = subprocess.run(
                    [sys.executable, "-X", "faulthandler", "-m", "src.application.worker",
                     str(directory / "request.json"), str(directory / "result.json")],
                    cwd=ROOT, env=env, stdout=log, stderr=log, timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Inference timed out; see outputs/logs/inference.log. Try CPU or a smaller image.") from exc
        if process.returncode:
            raise RuntimeError(f"Inference worker stopped (exit {process.returncode}); see outputs/logs/inference.log. Try CPU mode.")
        output = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if "error" in output:
            raise RuntimeError(output["error"] + " See outputs/logs/inference.log.")
        return output
