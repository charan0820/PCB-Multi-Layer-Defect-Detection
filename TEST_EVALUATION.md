# DeepPCB held-out test — 14 October 2026

The frozen Optical B/W YOLO11n detector was evaluated on the official **500-image,
3,140-region test split**, using the NVIDIA GeForce RTX 4050 Laptop GPU. No model
training, checkpoint selection or hyperparameter tuning used these test results.

## Overall results

| Metric | Held-out test | Previously recorded validation |
|---|---:|---:|
| Precision | 94.746% | 96.828% |
| Recall | 93.909% | 94.916% |
| mAP@50 | 96.664% | 98.465% |
| mAP@50–95 | 63.843% | 76.015% |

These are Ultralytics detector metrics, not a single accuracy score and not the
app's per-detection confidence. Precision and recall use Ultralytics' evaluation
operating point; they do not describe every possible application cutoff.

The stricter mAP@50–95 is about 12.17 percentage points below the previous
validation result. Detection remains strong at IoU 0.50, while precise box
localization is more difficult on this test split.

## Per-class results

| Class | Regions | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| open | 659 | 97.317% | 93.571% | 97.221% | 59.160% |
| short | 478 | 87.826% | 87.448% | 90.459% | 56.672% |
| mousebite | 586 | 97.137% | 94.881% | 97.641% | 61.631% |
| spur | 483 | 96.830% | 94.859% | 97.599% | 61.906% |
| copper | 464 | 98.226% | 95.460% | 98.199% | 74.214% |
| pin-hole | 470 | 91.143% | 97.234% | 98.864% | 69.476% |

Short has the lowest precision, recall and AP. Pin-hole has the next-lowest
precision. These observations are a final characterization of this frozen model;
do not repeatedly tune against this test set.

## Frozen protocol and integrity

- Checkpoint: `runs/deeppcb/yolo11n_baseline/weights/best.pt`.
- Checkpoint SHA-256: `5bca64f5f7baa7fcaa45a64ead6207014f63b718d26709400231869e2c145afd`.
- Manifest SHA-256: `2c2ab8daf2f51db8aa4722b4043269683085b368b512bcb1f2443492b49eaf65`.
- Resolution 640; batch 4; workers 0; device 0; seed 42; deterministic mode.
- Candidate confidence 0.001; NMS IoU 0.7; maximum detections 300.
- Augmentation off; half precision off; plots and prediction JSON enabled.
- Python 3.12.4; PyTorch 2.14.1+cu126; Ultralytics 8.4.175; CUDA 12.6.
- All 500 images were processed. Test image hashes matched the manifest and were
  distinct from recorded train/validation image hashes.
- Original test images, labels, manifest and checkpoint were unchanged after
  evaluation. Temporary copies were used to isolate generated label caches.
- Measured mean inference time: 3.641 ms/image, excluding preprocessing,
  postprocessing, startup and application/report overhead.

The first execution completed inference but failed while writing the report
because the script accessed an unavailable `model.validator` attribute. The
report writer was corrected to use the validator callback, then the evaluation
was repeated with the identical checkpoint hash and settings. No settings were
chosen from the first execution's scores. Both attempts remain in outputs.

Prediction JSON export also triggered a separate COCO evaluation. The table above
uses native Ultralytics box metrics, matching the earlier validation report;
do not mix them with the separate COCO evaluator's results. Ultralytics installed
faster-coco-eval 1.8.0 for this export; it is now included in the training
requirements so future environments can install it explicitly.

## Artifacts and reproducibility

Successful run: `outputs/reports/deeppcb_test_20261014_132353_755481/`.
It contains `evaluation_test.json`, `evaluation_test_per_class.csv`,
`protocol.json`, `test_inventory.json`, `predictions.json`, precision/recall
curves, confusion matrices and example annotated batches. Outputs are locally
ignored by Git; this summary records the durable result.

Command used:

```powershell
.venv\Scripts\python.exe scripts/evaluate_deeppcb.py
```

Keep the trained checkpoint, this report, protocol, inventory, predictions and
dataset provenance before archiving training data. Dataset-based reproducibility
requires restoring the original images and labels.

## Limits

The official split shares six source groups with training and one with validation.
It is image-held-out, **not board-disjoint**. This optical benchmark does not
establish real-camera performance, internal-layer detection or CT capability.
The original DeepPCB benchmark uses a different matching criterion; these scores
are not directly comparable with that benchmark.
