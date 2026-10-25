"""Evaluate the frozen DeepPCB checkpoint once on the official test split."""
import argparse
from collections import Counter
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inventory(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['classes'] != NAMES:
        raise ValueError('Dataset taxonomy does not match DeepPCB.')
    records = [r for r in manifest['records'] if r['split'] == 'test']
    if len(records) != 500 or manifest['splits']['test']['images'] != 500:
        raise ValueError('Expected the unchanged official 500-image test split.')
    images = dataset / 'images' / 'test'
    labels = dataset / 'labels' / 'test'
    if {p.name for p in images.glob('*.jpg')} != {r['image'] for r in records}:
        raise ValueError('Test images differ from the recorded split.')
    if {p.stem for p in labels.glob('*.txt')} != {Path(r['image']).stem for r in records}:
        raise ValueError('Missing or extra test labels.')
    other_hashes = {r['sha256'] for r in manifest['records'] if r['split'] != 'test'}
    test_hashes = set()
    counts = Counter()
    fingerprints = {}
    for record in records:
        name = record['image']
        if Path(name).name != name:
            raise ValueError('Test image must be a filename, not a path.')
        image, label = images / name, labels / (Path(name).stem + '.txt')
        image_hash = digest(image)
        if image_hash != record['sha256'] or image_hash in other_hashes or image_hash in test_hashes:
            raise ValueError(f'Changed or duplicate test image: {name}')
        test_hashes.add(image_hash)
        for line in label.read_text().splitlines():
            category, x, y, w, h = map(float, line.split())
            if not (category.is_integer() and 0 <= category < 6 and
                    0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f'Invalid annotation in {label.name}')
            counts[NAMES[int(category)]] += 1
        fingerprints[name] = {'image': image_hash, 'labels': digest(label)}
    groups = {r['group'] for r in records}
    overlap = {split: sorted(groups & set(manifest['splits'][split]['groups']))
               for split in ('train', 'val')}
    return manifest, counts, fingerprints, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/deeppcb/yolo11n_baseline/weights/best.pt')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/deeppcb_yolo')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    checkpoint, dataset = args.checkpoint.resolve(), args.dataset.resolve()
    checkpoint_hash = digest(checkpoint)
    manifest, counts, fingerprints, overlap = test_inventory(dataset)
    manifest_hash = digest(dataset / 'manifest.json')
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != NAMES:
        raise ValueError('Checkpoint taxonomy does not match DeepPCB.')
    validation_state = {}
    # The validator is supplied to callbacks; YOLO.val does not expose model.validator.
    model.add_callback('on_val_end', lambda validator: validation_state.update(seen=int(validator.seen)))
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly.')
    settings = dict(split='test', device=args.device, imgsz=640, batch=4, workers=0,
                    conf=0.001, iou=0.7, max_det=300, augment=False, half=False,
                    seed=42, deterministic=True, plots=True, save_json=True)
    output = ROOT / 'outputs' / 'reports' / f'deeppcb_test_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_hash, dataset=str(dataset),
                    manifest_sha256=manifest_hash, settings=settings, images=500,
                    instances=sum(counts.values()), class_instances=dict(counts),
                    source=manifest['source'], source_commit=manifest['source_commit'],
                    source_group_overlap=overlap, limitation=manifest['limitation'],
                    torch=str(torch.__version__), ultralytics=ultralytics_version,
                    cuda=torch.version.cuda,
                    hardware='CPU' if args.device == 'cpu' else torch.cuda.get_device_name(0))
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'test_inventory.json').write_text(json.dumps(fingerprints, indent=2), encoding='utf-8')
    print(f'Frozen checkpoint SHA-256: {checkpoint_hash}', flush=True)
    print(f'Test set: 500 images, {sum(counts.values())} regions; device: {metadata["hardware"]}', flush=True)
    print(f'Output: {output}', flush=True)
    # Evaluate copies: Ultralytics may create label caches; source data stays untouched.
    with tempfile.TemporaryDirectory(prefix='test-staging-', dir=output) as directory:
        staging = Path(directory)
        for kind in ('images', 'labels'):
            shutil.copytree(dataset / kind / 'test', staging / kind / 'test')
        local = staging / 'test.yaml'
        local.write_text(yaml.safe_dump(dict(path=str(staging), train='images/test',
                         val='images/test', test='images/test', names=dict(enumerate(NAMES))),
                         sort_keys=False), encoding='utf-8')
        metrics = model.val(data=str(local), project=str(output.parent), name=output.name,
                            exist_ok=True, **settings)
    if validation_state.get('seen') != 500:
        raise RuntimeError('Evaluation did not cover all 500 test images.')
    unchanged = digest(checkpoint) == checkpoint_hash and digest(dataset / 'manifest.json') == manifest_hash
    for name, hashes in fingerprints.items():
        unchanged = unchanged and digest(dataset / 'images/test' / name) == hashes['image']
        unchanged = unchanged and digest(dataset / 'labels/test' / (Path(name).stem + '.txt')) == hashes['labels']
    if not unchanged:
        raise RuntimeError('Checkpoint or original test data changed during evaluation.')
    box = metrics.box
    per_class = []
    for index, category in enumerate(box.ap_class_index):
        name = NAMES[int(category)]
        per_class.append(dict(class_name=name, instances=counts[name],
                              precision=float(box.p[index]), recall=float(box.r[index]),
                              map50=float(box.ap50[index]), map50_95=float(box.ap[index])))
    payload = {**metadata, 'source_files_unchanged': True,
               'metrics': dict(precision=float(box.mp), recall=float(box.mr),
                               map50=float(box.map50), map50_95=float(box.map)),
               'per_class': per_class, 'speed_ms_per_image': metrics.speed}
    (output / 'evaluation_test.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with (output / 'evaluation_test_per_class.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    print(json.dumps(payload['metrics'], indent=2), flush=True)
    print(f'Completed held-out test: {output / "evaluation_test.json"}', flush=True)


if __name__ == '__main__':
    main()
