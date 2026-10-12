"""Full-dataset conversion fidelity, relocation and taxonomy regression tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
from PIL import Image
from src.application.inspection import PCB_CLASSES, DEEPPCB_CLASSES, is_pcb_model
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/deeppcb_yolo'

def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_all_annotations_and_pairs():
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert {k:v['images'] for k,v in manifest['splits'].items()} == {'train':800,'val':200,'test':500}
    assert not set(manifest['splits']['train']['groups']) & set(manifest['splits']['val']['groups'])
    keys,hashes=set(),set()
    for r in manifest['records']:
        split=r['split'];image=DATA/'images'/split/r['image']
        assert r['id'] not in keys
        keys.add(r['id'])
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        assert digest == r['sha256'] and digest not in hashes
        hashes.add(digest)
        with Image.open(image) as im:
            W,H=im.size
            assert (W,H)==(r['width'],r['height'])
        with Image.open(DATA/'templates'/split/r['template']) as im:
            assert im.size == (W,H)
        original=(DATA/'original_annotations'/split/(r['id']+'.txt')).read_text().splitlines()
        converted=(DATA/'labels'/split/(image.stem+'.txt')).read_text().splitlines()
        assert len(original)==len(converted)>0
        for raw,yolo in zip(original,converted):
            x1,y1,x2,y2,c=map(int,raw.split())
            category,x,y,w,h=map(float,yolo.split())
            assert category==c-1
            assert [(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H] == pytest.approx([x1,y1,x2,y2],abs=.0001)
    assert len(keys)==1500

def test_portable_preflight(tmp_path):
    relocated=tmp_path/'dataset with spaces';relocated.mkdir()
    for name in ['manifest.json','deeppcb.yaml']:
        shutil.copy2(DATA/name,relocated/name)
    for name in ['images','labels']:
        try:
            (relocated/name).symlink_to(DATA/name,target_is_directory=True)
        except OSError:
            # Windows commonly requires Developer Mode/admin for symlinks.
            shutil.copytree(DATA/name, relocated/name)
    path=script('train_deeppcb').validate_dataset(relocated)
    import yaml
    assert yaml.safe_load(path.read_text())['train']==str(relocated/'images/train')

@pytest.mark.parametrize('bad',['0 0 10 10 0','0 0 10 10 7','0 0 0 10 1','-1 0 10 10 1','0 0 641 10 1','0 0 10 10'])
def test_invalid_boxes(tmp_path,bad):
    label=tmp_path/'bad.txt';label.write_text(bad)
    with pytest.raises(ValueError):
        script('prepare_deeppcb').convert_boxes(label,640,640)

def test_taxonomy():
    assert is_pcb_model(list(PCB_CLASSES)) and is_pcb_model(list(DEEPPCB_CLASSES))
    assert not is_pcb_model(['person']*6)
    assert not is_pcb_model(['open','short','mousebite','spur','copper','missing_hole'])
