"""Windows-safe, single-GPU DeepPCB training with RTX 4050 defaults."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']


def validate_dataset(dataset):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    config = yaml.safe_load((dataset / 'deeppcb.yaml').read_text())
    if list(config['names'].values()) != NAMES or manifest['classes'] != NAMES:
        raise ValueError('Dataset classes do not match DeepPCB')
    for split in ['train', 'val', 'test']:
        images = sorted((dataset / 'images' / split).glob('*.jpg'))
        labels = sorted((dataset / 'labels' / split).glob('*.txt'))
        if len(images) != manifest['splits'][split]['images'] or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f'{split}: missing or extra images/labels')
        for label in labels:
            for line in label.read_text().splitlines():
                category, x, y, w, h = map(float, line.split())
                if not (category.is_integer() and 0 <= category <= 5 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'Invalid YOLO box in {label}')
    # Absolute paths are generated locally, so moving/extracting the ZIP is safe.
    resolved = dict(train=str(dataset/'images/train'), val=str(dataset/'images/val'),
                    test=str(dataset/'images/test'), names=dict(enumerate(NAMES)))
    target = dataset / 'deeppcb.local.yaml'
    target.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/deeppcb_yolo')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--smoke', action='store_true', help='Three epochs before full training')
    p.add_argument('--check-only', action='store_true', help='Validate dataset without importing torch')
    p.add_argument('--evaluate', type=Path, help='Evaluate specified best.pt on untouched test split')
    p.add_argument('--resume', type=Path, help='Resume interrupted training from last.pt')
    args = p.parse_args()
    data = validate_dataset(args.dataset)
    print(f'Dataset validated: {data}')
    if args.check_only:
        return
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch and verify your NVIDIA driver. CPU training was not started.')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}; VRAM: {props.total_memory/1024**3:.1f} GiB; batch: {args.batch}')
    if args.batch < 1 or args.epochs < 1:
        p.error('batch and epochs must be positive')
    if args.evaluate:
        model = YOLO(str(args.evaluate.resolve()))
        if list(model.names.values()) != NAMES:
            raise ValueError('Checkpoint classes/order do not match DeepPCB')
        model.val(data=str(data), split='test', device=0, imgsz=640, batch=args.batch,
                  workers=args.workers, project=str(ROOT/'runs/deeppcb'), name='test_evaluation')
        return
    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    if args.resume:
        model.train(resume=True, device=0, workers=args.workers)
    else:
        model.train(data=str(data), epochs=3 if args.smoke else args.epochs,
                    batch=args.batch, imgsz=640, patience=20, device=0, amp=True,
                    workers=args.workers, cache=False, seed=42,
                    project=str(ROOT/'runs/deeppcb'), name='smoke_test' if args.smoke else 'yolo11n_baseline',
                    exist_ok=False, hsv_h=0.0, hsv_s=0.0)
    # Trainer is authoritative when Ultralytics increments an existing run directory.
    save_dir = Path(model.trainer.save_dir)
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), torch=str(torch.__version__),
                    ultralytics=ultralytics_version, cuda=torch.version.cuda, gpu=props.name,
                    dataset=str(data), options=vars(args))
    (save_dir/'environment.json').write_text(json.dumps(metadata, default=str, indent=2))
    print(f'Training complete. Best weights: {save_dir / "weights/best.pt"}')
    print('In the app sidebar, select this checkpoint. Evaluate the test split only after selecting your final model.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
