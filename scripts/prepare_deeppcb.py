"""Convert official DeepPCB pairs to validated YOLO data; no ML dependency."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']


def read_split(source, filename):
    records = []
    for line in (source / filename).read_text().splitlines():
        if not line.strip():
            continue
        image_ref, label_ref = line.split()
        base = source / image_ref
        image = base.with_name(base.stem + '_test.jpg')
        template = base.with_name(base.stem + '_temp.jpg')
        label = source / label_ref
        for path in [image, template, label]:
            if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
                raise ValueError(f'Missing or unsafe dataset file: {path}')
        records.append(dict(image=image, template=template, label=label,
                            group=Path(image_ref).parts[0], key=base.stem))
    keys = [r['key'] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate image IDs in {filename}')
    return records


def convert_boxes(label, width, height):
    boxes, classes = [], []
    for number, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = re.split(r'[\s,]+', line.strip())
        if len(values) != 5:
            raise ValueError(f'{label}:{number}: expected five integers')
        x1, y1, x2, y2, category = map(int, values)
        if not (1 <= category <= 6 and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'{label}:{number}: invalid class or bounding box')
        boxes.append(f'{category-1} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
        classes.append(category-1)
    if not boxes:
        raise ValueError(f'Unexpected empty defect annotation: {label}')
    return '\n'.join(boxes) + '\n', classes


def prepare(source, output, val_fraction=.2):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (source / 'PCBData').is_dir():
        source /= 'PCBData'
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new output folder; existing data is never overwritten.')
    trainval, test = read_split(source, 'trainval.txt'), read_split(source, 'test.txt')
    if len(trainval) != 1000 or len(test) != 500:
        raise ValueError('Expected official 1000 trainval / 500 test records')
    if {r['key'] for r in trainval} & {r['key'] for r in test}:
        raise ValueError('Train/test image ID overlap')
    counts = Counter(r['group'] for r in trainval)
    groups = sorted(counts)
    choices = (c for n in range(1, len(groups)) for c in itertools.combinations(groups, n))
    selected = min(choices, key=lambda c: (abs(sum(counts[g] for g in c)-len(trainval)*val_fraction), len(c), c))
    splits = {'train': [r for r in trainval if r['group'] not in selected],
              'val': [r for r in trainval if r['group'] in selected], 'test': test}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='deeppcb-convert-', dir=output.parent))
    manifest = dict(source='https://github.com/tangsanli5201/DeepPCB',
                    classes=NAMES, validation_groups=list(selected), splits={}, records=[])
    seen_hashes = {}
    try:
        for split, records in splits.items():
            for directory in ['images', 'labels', 'templates', 'original_annotations']:
                (staging / directory / split).mkdir(parents=True)
            class_counts = Counter()
            for r in records:
                with Image.open(r['image']) as im:
                    im.load()
                    width, height = im.size
                with Image.open(r['template']) as im:
                    im.load()
                    if im.size != (width, height):
                        raise ValueError(f'Template size mismatch: {r["key"]}')
                text, classes = convert_boxes(r['label'], width, height)
                digest = hashlib.sha256(r['image'].read_bytes()).hexdigest()
                if digest in seen_hashes:
                    raise ValueError(f'Duplicate image content: {r["key"]} and {seen_hashes[digest]}')
                seen_hashes[digest] = r['key']
                name = r['image'].stem
                shutil.copy2(r['image'], staging / 'images' / split / r['image'].name)
                shutil.copy2(r['template'], staging / 'templates' / split / r['template'].name)
                shutil.copy2(r['label'], staging / 'original_annotations' / split / r['label'].name)
                (staging / 'labels' / split / (name + '.txt')).write_text(text)
                class_counts.update(classes)
                manifest['records'].append(dict(id=r['key'], split=split, group=r['group'],
                                               image=r['image'].name, template=r['template'].name,
                                               sha256=digest, width=width, height=height))
            manifest['splits'][split] = dict(images=len(records), groups=sorted({r['group'] for r in records}),
                                           defects={NAMES[k]: class_counts[k] for k in range(6)})
            if set(class_counts) != set(range(6)):
                raise ValueError(f'{split} does not cover all six classes')
        # Relative to this YAML's directory: portable after ZIP extraction.
        config = dict(train='images/train', val='images/val', test='images/test',
                      names=dict(enumerate(NAMES)))
        (staging / 'deeppcb.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest['limitation'] = ('Validation groups are disjoint from training. Official test split is preserved, '
                                 'but shares some source groups with training/validation; not a board-disjoint test.')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        previews = staging / 'previews'
        previews.mkdir()
        for split, records in splits.items():
            for r in records[:3]:
                with Image.open(r['image']) as im:
                    image = im.convert('RGB')
                draw = ImageDraw.Draw(image)
                for line in (staging / 'labels' / split / (r['image'].stem+'.txt')).read_text().splitlines():
                    c, x, y, w, h = map(float, line.split())
                    W, H = image.size
                    box = ((x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H)
                    draw.rectangle(box, outline='red', width=2)
                    draw.text((box[0], max(0, box[1]-12)), NAMES[int(c)], fill='red')
                image.save(previews / f'{split}_{r["key"]}.png')
        shutil.move(str(staging), str(output))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='DeepPCB checkout or PCBData directory')
    p.add_argument('--output', type=Path, default=ROOT/'data/deeppcb_yolo')
    p.add_argument('--val-fraction', type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.val_fraction < 1:
        p.error('val-fraction must be between zero and one')
    manifest = prepare(args.source, args.output, args.val_fraction)
    print(json.dumps(manifest['splits'], indent=2))
    print(f'Ready: {args.output / "deeppcb.yaml"}')


if __name__ == '__main__':
    main()
