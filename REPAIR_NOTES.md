# Repair verification

Baseline inspected: Git commit `ef07065` from the supplied GitHub repository.

## Confirmed problems in the code

- Model selection silently fell back to generic YOLO weights when the trained checkpoint was missing.
- Upload cache identity used filename/size and omitted the checkpoint identity. Different bytes and changed models could reuse stale detections.
- Every threshold rerun rendered with Matplotlib inside the Streamlit process. Native inference also lived in that process, so a native fault could stop the application with no Python exception.
- The download advertised a PNG payload using the uploaded filename extension; it was an annotated image rather than an inspection report.
- YOLO load exceptions were swallowed before attempting an unrelated classifier fallback.

The original Windows silent crash was not reproduced here. There is no crash dump or Windows/GPU environment available. These are observed weaknesses and mitigations, not a proven diagnosis of a specific native fault.

## Verification performed

- `python -m pytest tests/ -q`: **14 passed, 2 skipped**. Dataset-only checks skip because the raw dataset is absent; they run if it is installed.
- Streamlit AppTest: injected upload, inspection, four confidence threshold reruns and deliberate rescan. Threshold changes produced no extra inference or database entries. Explicit rescan created another snapshot while reusing raw inference.
- Download widgets: HTML, JSON and PNG use correct MIME/extensions and the no-rerun flag. Report bytes were tested for current threshold, JSON consistency, HTML escaping and valid PNG dimensions. A real browser download click was not automated.
- Image validation, content/model cache identity, idempotent SQLite writes, and worker abnormal exits/timeouts covered by regression tests.
- Real isolated worker runs for **both attached checkpoints**, with synthetic images and repeated threshold rendering: inference completed on CPU. Each has **80 general object classes**. The smoke tool returns code 2 deliberately after a successful execution check because these weights are not PCB detectors.
- Streamlit startup and live health endpoint: **HTTP 200 / ok**.
- Real generic checkpoint rejection in Streamlit: reported 80 classes and disabled inspection, without an uncaught exception.
- Verified UI service imports do not import PyTorch or Matplotlib.
- `git diff --check`: passed.

Validated environment: Linux, Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.175, Streamlit 1.65.0, Pillow 12.3.0.

## Remaining acceptance work

Provide the PCB-trained `best.pt` and representative PCB images. Run the smoke tool and manual acceptance sequence in README on your Windows machine, first CPU then CUDA. Real PCB detection quality, end-to-end physical hardware behavior and the original Windows native crash are not verified by synthetic inputs or mocked UI predictions.

The report/database/UI are part of the current optical prototype. No second model, CT implementation, hardware acquisition or retraining was started. Historical evaluation JSON was retained but has not been reproduced or tied to the attached generic checkpoints.
