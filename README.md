# Multilayer PCB Defect Detector

The application has three modes: **Surface**, **Optical B/W** and **CT scans**.
Surface uses the original surface detector; Optical B/W uses the separate DeepPCB
detector. Each optical mode retains its own checkpoint, device, image size, IoU and
confidence settings. CT scans displays **Not configured** until a separate model
and CT processing pipeline are ready. CT and camera hardware are not implemented.

Reports record the selected mode and its scope, and filenames include the mode.
SQLite statistics filter by mode. Existing scans without mode metadata remain in
the database and are excluded from mode-specific totals. Checkpoints with the wrong
mode's taxonomy and generic pretrained models are rejected.

See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the completed local DeepPCB run.

## Frontend theme

The inspection workspace uses solid solder-mask greens, copper pads and
circuit routing artwork. A horizontal configuration bar sits above a dominant
board canvas and right-hand image intake panel. On smaller screens, the intake
panel moves ahead of the canvas. Locally bundled Barlow Condensed headings and
Barlow body text establish the typography; their license is in `app/assets/fonts/OFL.txt`.
Model configuration stays under **Model & advanced settings**. Tab changes and
control states use eased transitions with reduced-motion support.
Colour follows an approximate 60–30–10 hierarchy: deep board green for the
dominant background, lighter green for workspace surfaces, and restrained
copper/signal-green accents for actions, navigation and board details.
The latest composition follows the supplied Wix wh-1081 reference: a large,
light sans-serif title over decorative PCB wireframes, simple anchor navigation,
square controls, fine dividers, and a wide copper link into the inspection workspace.

Theme defaults live in `.streamlit/config.toml`; reusable presentation styles
and helpers live in `app/studio.css` and `app/theme.py`. Circuit entrance motion
and interaction transitions respect the system's reduced-motion preference.
Restart Streamlit after editing the theme configuration or Python helpers
(file watching is disabled), then reload the browser.

## DeepPCB training

Start with [START_TRAINING.md](START_TRAINING.md). Includes dataset conversion, paired-image retention, provenance, GPU training, resume and held-out test evaluation. The app accepts DeepPCB classes `open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole` as well as the original taxonomy.

## Storage and dataset retention

Uploaded-image inference does not require training datasets. Keep the trained
checkpoints at `runs/pcb_defect/baseline/weights/best.pt` and
`runs/deeppcb/yolo11n_baseline/weights/best.pt`, along with the application source,
configuration, fonts and Python environment. Keep `outputs/inspection.sqlite3`
if you want to retain scan history.

`data/raw/pcb-defect-dataset` (about 1.11 GiB locally) and `data/deeppcb_yolo`
(about 121 MiB locally) may be archived or removed for an inference-only setup.
Their removal disables the local validation-sample selector for the respective
mode and prevents dataset-based evaluation, retraining and dataset integration
tests until the data is restored. Preserve dataset manifests, split information,
source terms and evaluation reports when archiving for reproducibility.

The final DeepPCB held-out test evaluation is complete; see
[TEST_EVALUATION.md](TEST_EVALUATION.md) for overall/per-class scores and the frozen
protocol. Archive the data and provenance if future reproduction is important.
Delete dataset directories only, not the entire `data` folder: it also contains
tracked Python source and the dataset YAML.

Project cleanup removed empty staging directories, placeholder notebooks,
an identical duplicate pretrained-weight folder, generated frontend screenshots,
and Python/test caches. Active training runs, trained checkpoints, evaluation
evidence and datasets were retained. Caches and output directories are recreated
by their tools when needed.

## Run the repaired application

Use Python 3.10–3.12. From the extracted project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m streamlit run app/app.py --server.fileWatcherType none
```

On Linux/macOS activate with `source .venv/bin/activate`.
For CPU-only PyTorch, install `torch torchvision --index-url https://download.pytorch.org/whl/cpu` before the requirements. For CUDA use the appropriate PyTorch installation for your GPU. CPU is the default during crash diagnosis.

**Checkpoints:** Surface defaults to `runs/pcb_defect/baseline/weights/best.pt`;
Optical B/W defaults to `runs/deeppcb/yolo11n_baseline/weights/best.pt` and opens by
default. Enter the actual full-run path if training created an incremented folder.
Trained weights are local artifacts and are not included in Git. Generic
`yolo11n.pt` / `yolo26n.pt` are pretrained starting weights and are rejected for
inspection. The converted DeepPCB dataset is bundled; follow
[START_TRAINING.md](START_TRAINING.md) for RTX 4050 training and validation review.

Expected checkpoint class names (order is taken from the checkpoint): `mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, `spurious_copper`.

## How it works

1. Assign a PCB/batch ID and upload a JPEG or PNG (maximum 15 MB / 20 megapixels). EXIF orientation is applied; images become RGB. Ultralytics performs its own aspect-preserving letterboxing; the application does not stretch the image or apply training augmentation.
2. Press **Inspect PCB**. Inference runs with a 0.01 candidate cutoff, configured image size and IoU. CPU is the stable default; CUDA GPU 0 is optional.
3. Retained detections are filtered at the visible cutoff, drawn with Pillow and displayed beside the original image. Moving the slider never runs inference again.
4. Download an annotated PNG, structured JSON or a complete standalone HTML report. Open the HTML in a browser and print/save as PDF. A report contains PCB ID, scan ID/time, source image hash (JSON), checkpoint identity, runtime, cutoff, per-region confidence and pixel coordinates plus annotated image.
5. SQLite stores each explicit inspection once. Scan history shows counts by time period/hour and defect type, flagged scans and mean detection confidence. It refreshes every 10 seconds.

History stores the cutoff selected when **Inspect PCB** was pressed. Slider previews/downloads reflect the current cutoff and do not alter history. Pressing Inspect again deliberately creates another scan snapshot; unchanged image/model settings reuse raw inference. Metadata edits take effect on the next explicit inspection. Original pixels and annotated images are kept only in the current browser session; the database retains metadata and detections, not image blobs.

## Crash repair

- Streamlit no longer loads PyTorch, CUDA or Matplotlib in its application process. Inference lives in a bounded subprocess; a worker access violation or timeout becomes an error message while the UI remains running. Each new image/model loads its model in the worker, so startup overhead is higher than a persistent GPU service.
- Content SHA-256, checkpoint path/size/mtime, device, image size and IoU identify the cached candidates. Same-named/same-sized uploads no longer share results; changing checkpoints cannot silently reuse the previous model's predictions.
- Downloads use Streamlit's `on_click="ignore"` and precomputed bytes. PNG exports always have a `.png` extension. HTML filename/text metadata is escaped.
- Matplotlib is kept only for the existing offline visualization functions; the app draws native-size overlays with Pillow. Slider reruns replace one rendered view instead of accumulating figures.
- No automatic fallback to generic pretrained weights. Original YOLO load errors are preserved instead of silently trying an incompatible classifier.
- Watchers are disabled in `.streamlit/config.toml` and via the launch command. Worker, application and native fault logs are in `outputs/logs/`.

These changes address failure-prone paths. The original silent Windows crash cannot be attributed to a specific DLL without its logs/dump. Validation was performed on Linux CPU; repeat the acceptance steps on the Windows/GPU setup before claiming that environment is fixed.

## Verification

```powershell
python -m pytest tests/ -q
python scripts/smoke_checkpoint.py runs/pcb_defect/baseline/weights/best.pt --image path/to/pcb.jpg
```

The smoke command runs the real checkpoint once, changes the cutoff repeatedly and renders PNGs. Generic weights exit with code 2 after execution, clearly indicating they cannot be used for PCB inspection. A blank synthetic image is used if `--image` is omitted; it checks execution, not accuracy.

Manual acceptance: upload a real PCB image, inspect, move the cutoff both directions, download all three formats repeatedly, and inspect a different image with the same filename. The server must remain alive, reports must match the preview, and history must not increment for slider changes/downloads. Then repeat with GPU selected if CUDA is installed. If failure occurs, retain `outputs/logs/app.log`, `app-native.log` and `inference.log`.

The raw dataset is optional for application tests. Two existing dataset integration tests skip when the dataset is absent; installing it enables them. No dataset, trained PCB checkpoint or original Windows crash dump was available for this repair. The uploaded generic models were exercised successfully on CPU. See `REPAIR_NOTES.md` for exact results.

## Statistics and evaluation

Detection confidence is not accuracy. Unlabeled uploads cannot yield a live accuracy score. The UI explicitly marks accuracy unavailable. The repository's historical test JSON reports precision 83.2%, recall 71.8%, mAP@50 80.3% and mAP@50–95 36.0%; those figures have not been reproduced in this repair or linked to the supplied checkpoints. A no-detection result does not prove a defect-free board.

The existing training/evaluation scripts remain available (`scripts/train_yolo.py`, `scripts/evaluate_yolo.py`). Training is not part of this repair.

## Project structure

- `app/app.py`: Streamlit UI, explicit scan action, state and downloads.
- `src/application/inspection.py`: upload validation, checkpoint identity and worker orchestration.
- `src/application/worker.py`: isolated model loading/inference.
- `src/application/reports.py`: Pillow overlay, printable HTML and JSON.
- `src/application/history.py`: transactional SQLite scan history/statistics.
- `src/inference/predict.py`: inference and pure-Python candidate filtering.
- `src/models/model_loader.py`: checkpoint loading/saving.
- Existing `src/data`, `src/training`, `src/evaluation`, `src/visualization`: dataset/training/offline utilities.
- `tests/test_application.py`: rerun, report, history, invalid image and worker failure regression tests.
- `.streamlit/config.toml`, `config.yaml`: watcher/theme and inference/database configuration.

## Future optical setup for presentation only

Place the PCB flat in a repeatable jig within a matte enclosure. Mount a close-focus USB camera vertically above it; surround the lens with a diffuse ring light to reduce glare. Choose magnification after measuring the smallest defect you need to see; pixel coverage and repeatable focus matter more than nominal camera megapixels. Use a suitable regulated supply for the light. Connect the camera by USB to the laptop running this app; a saved JPEG/PNG enters the current upload pipeline. Validate on labeled images from this setup before claiming camera performance. No hardware integration is included.

DeepPCB contains paired optical template/test images captured with a linear-scan CCD, not CT scans: https://github.com/tangsanli5201/DeepPCB. The baseline uses the defective image alone. A future CT model needs suitable CT data and a separate acquisition/validation plan. PCB-CTSeg is a proposed structural segmentation dataset for pads, traces and vias; its labels do not establish manufacturing defect detection.
