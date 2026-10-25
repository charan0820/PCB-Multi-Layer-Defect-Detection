# Start DeepPCB training — RTX 4050, 6 GB

The ZIP includes the repaired app plus the **already converted complete DeepPCB dataset**.
You do not need to download or convert it again. No trained DeepPCB checkpoint is included.
Train locally on your Windows laptop; the packaging environment did not run GPU training.

## 1. Install dependencies once

Install Python **3.11 64-bit** and an up-to-date NVIDIA driver.
Extract the ZIP to a short path, e.g. `C:\PCB_Project`. Open Command Prompt inside
its `PCB-DeepPCB-Ready` folder (the folder containing this document).

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

At https://pytorch.org/get-started/locally/ choose Stable, Windows, Pip, Python,
and a CUDA build supported by your driver. Run the generated install command in
this activated environment. The PyTorch wheel brings its CUDA runtime; a separate
CUDA Toolkit/cuDNN install is not required for this workflow.

```bat
python -m pip install -r requirements-training.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'STOP: GPU not available')"
python scripts\train_deeppcb.py --check-only
```

CUDA must be available and the GPU name should be RTX 4050. Plug in the laptop,
use performance mode, and close the PCB app and other GPU workloads.

## 2. Start the three-epoch smoke test

```bat
TRAIN_DEEPPCB.bat --smoke
```

This tests the full dataset loading, GPU training, validation and checkpoint path.
It uses YOLO11n, image size 640, batch 4, mixed precision and workers 0.
On first use Ultralytics downloads the general pretrained `yolo11n.pt` automatically;
this is valid for transfer learning. Alternatively copy your supplied `yolo11n.pt`
to this folder. Generic weights are not a finished PCB detector.
If CUDA runs out of memory, retry with `TRAIN_DEEPPCB.bat --smoke --batch 2`, then 1.
Other errors should be diagnosed rather than lowering batch indiscriminately.

## 3. Start full training after the smoke test succeeds

```bat
TRAIN_DEEPPCB.bat
```

Defaults: up to 100 epochs, patience 20, batch 4, seed 42. Use `--batch 2`
if required by the smoke test. Run folders automatically increment on reruns,
so existing checkpoints are not overwritten. Color hue/saturation augmentation
is disabled for the binary/grayscale source images; other YOLO defaults apply.

Expected first full-run checkpoint:
`runs/deeppcb/yolo11n_baseline/weights/best.pt`.
The launcher prints the actual path. Review per-class validation precision,
recall, mAP50 and mAP50–95; confidence is not measured accuracy.

If training is interrupted, resume the actual run's last checkpoint:

```bat
python scripts\train_deeppcb.py --resume runs\deeppcb\yolo11n_baseline\weights\last.pt
```

Resume restores saved run settings, including batch/epochs; it is not a new run.
After choosing the final checkpoint using validation only, evaluate the test split:

```bat
python scripts\evaluate_deeppcb.py --checkpoint runs\deeppcb\yolo11n_baseline\weights\best.pt
```

The evaluation script verifies the official 500-image split and writes a frozen
protocol, integrity inventory, JSON/CSV scores and plots under `outputs/reports/`.

Do not repeatedly tune on the test results. Dataset tests use standard YOLO metrics;
DeepPCB's original benchmark uses a different IoU criterion, so published scores
are not directly comparable.

## 4. Open the repaired app with the trained checkpoint

```bat
python -m pip install -r requirements.txt
python -m streamlit run app/app.py --server.fileWatcherType none
```

Select **Optical B/W** and enter the printed full-run `best.pt` path in
**Trained checkpoint**. Surface has its own checkpoint/settings and requires the
original surface taxonomy. CT scans displays **Not configured**.
The app supports the original PCB taxonomy and DeepPCB's taxonomy separately.
A pin hole is not relabeled as a missing hole. Choose a bundled local test sample
or upload a PCB image. Optical templates are retained but this YOLO baseline uses
only the test image, not a template comparison or CT model.

Before final test evaluation, review the completed run on validation:

```bat
python scripts\review_deeppcb.py --checkpoint runs\deeppcb\yolo11n_baseline\weights\best.pt
```

This writes a checkpoint-hashed JSON review, per-class metrics and plots under
`outputs/reviews/` in a new folder. Temporary validation copies keep the source
dataset and checkpoints untouched. It does not evaluate the test split. The app's
local samples also use validation images so the final test is reserved for evaluation.

## Dataset preparation and provenance

Source: https://github.com/tangsanli5201/DeepPCB
Source commit: `08e98c4db5922613fb97176eb3d6497d48260cb1`.
The source README states research-only dataset use; source README/LICENSE are
included under `data/deeppcb_yolo/source_terms`. Cite the dataset/paper in your report.

- **800 train / 200 validation / 500 test** images, each with its original template.
- The official 500-image test split is preserved exactly.
- Validation holds out `group00041` from the official training pool; it shares no
  source group with the new training split.
- The official test split shares some source groups with train/validation. These
  results are not evidence of generalization to entirely unseen boards.
- Six YOLO classes: `0 open`, `1 short`, `2 mousebite`, `3 spur`, `4 copper`, `5 pin-hole`.
- Original 1-based class IDs become 0-based IDs. Bounding boxes become normalized
  center/width/height coordinates using actual image dimensions, without changing pixels.
- Original annotation files, pair mapping, image SHA-256 hashes, class counts and
  group assignments are retained in `manifest.json` and `original_annotations`.
- Missing files, corrupt images, invalid boxes and duplicate image content are rejected.
- Inspect `data/deeppcb_yolo/previews` for example converted box overlays.
- Raw DeepPCB contains artificially added defects and binary CCD imagery. Validate
  separately on your camera images before claiming hardware inspection accuracy.

To regenerate from a fresh official checkout into a new folder:

```bat
python scripts\prepare_deeppcb.py --source C:\datasets\DeepPCB --output data\deeppcb_regenerated
python scripts\train_deeppcb.py --dataset data\deeppcb_regenerated --check-only
```

The converter refuses to overwrite an existing output folder. The trainer writes
`deeppcb.local.yaml` with paths resolved on your machine; moving the ZIP is supported.
