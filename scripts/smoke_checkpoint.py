"""Exercise real isolated inference; does not measure accuracy or train anything."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from src.application.inspection import PCB_CLASSES, decode_image, resolve_checkpoint, run_worker
from src.application.reports import render_overlay
from src.inference.predict import filter_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    image = decode_image(args.image.read_bytes()) if args.image else Image.new("RGB", (640, 480))
    output = run_worker(checkpoint, image, device=args.device)
    print(f"Checkpoint: {checkpoint.name}; classes: {len(output['names'])}; device: {args.device}")
    for cutoff in (.05, .25, .5, .95, .25):
        result = filter_detections(output["prediction"], cutoff)
        overlay = render_overlay(image, result)
        print(f"Cutoff={cutoff:.2f}; regions={result['defect_count']}; PNG bytes={len(overlay)}")
    print("Inference and repeated threshold rendering succeeded.")
    if len(output["names"]) != 6 or frozenset(output["names"]) != PCB_CLASSES:
        print("NOT A PCB CHECKPOINT: the dashboard will reject these weights. Supply trained best.pt.")
        return 2
    if not args.image:
        print("Synthetic blank input: this confirms execution only, not detection quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
