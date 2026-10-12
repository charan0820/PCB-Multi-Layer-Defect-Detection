"""Review a completed DeepPCB checkpoint on validation, preserving source data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/deeppcb/yolo11n_baseline/weights/best.pt')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/deeppcb_yolo')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    checkpoint, dataset = args.checkpoint.resolve(), args.dataset.resolve()
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest['classes'] != NAMES:
        raise ValueError('Dataset taxonomy does not match DeepPCB')
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != NAMES:
        raise ValueError('Checkpoint taxonomy does not match DeepPCB')
    rows = list(csv.DictReader((checkpoint.parents[1]/'results.csv').open()))
    best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
    output = ROOT/'outputs/reviews'/f'deeppcb_validation_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    # Ultralytics may create label caches. Copies keep the original dataset untouched.
    with tempfile.TemporaryDirectory(prefix='deeppcb-review-') as directory:
        staging = Path(directory)
        for kind in ('images', 'labels'):
            shutil.copytree(dataset/kind/'val', staging/kind/'val')
        local = staging/'validation.yaml'
        local.write_text(yaml.safe_dump(dict(path=str(staging), train='images/val', val='images/val',
                                             names=dict(enumerate(NAMES))), sort_keys=False))
        metrics = model.val(data=str(local), split='val', device=args.device, imgsz=640, batch=4,
                            workers=0, project=str(output.parent), name=output.name,
                            exist_ok=True, plots=True)
    payload = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   created_at=datetime.now(timezone.utc).isoformat(), split='val',
                   images=manifest['splits']['val']['images'], epochs_completed=len(rows),
                   best_training_row=best, validation_metrics=metrics.results_dict,
                   per_class=metrics.summary(), speed=metrics.speed,
                   limitation=manifest['limitation'], test_evaluated=False)
    target = output/'review.json'
    target.write_text(json.dumps(payload, indent=2, default=lambda value: value.item()), encoding='utf-8')
    print(f'Validation review: {target}')
    print('The official test split was not evaluated. Freeze checkpoint/settings before final evaluation.')


if __name__ == '__main__':
    main()
