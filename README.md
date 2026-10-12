# PCB Surface Inspection Prototype

This application detects optical PCB surface defects using an existing trained Ultralytics YOLO checkpoint. It does not perform CT, X-ray or internal multilayer inspection. This repair does not train a second model or implement camera hardware.

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

**Required checkpoint:** copy your PCB-trained `best.pt` to `runs/pcb_defect/baseline/weights/best.pt`, or enter its local path in the sidebar. The uploaded `yolo11n.pt` and `yolo26n.pt` each contain 80 general object classes, not the six PCB classes. They are intentionally rejected by the dashboard. No weights or datasets are bundled in this delivery.

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

DeepPCB contains paired optical template/test images captured with a linear-scan CCD, not CT scans: https://github.com/tangsanli5201/DeepPCB. Training on DeepPCB alone does not enable internal-layer inspection. A future CT model needs suitable CT data and a separate acquisition/validation plan. The second model is deferred until this prototype passes acceptance with the trained PCB checkpoint.
