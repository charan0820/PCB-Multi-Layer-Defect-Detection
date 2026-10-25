# Completed DeepPCB validation review — 12 October 2026

Full run: `runs/deeppcb/yolo11n_baseline`, YOLO11n, RTX 4050 Laptop GPU,
batch 4, 640 pixels, seed 42, workers 0. The completion marker
`environment.json` exists. Training recorded 89 epochs of a maximum 100;
the best CSV row is epoch 69, consistent with stopping after patience 20.
Saved `best.pt` metrics match that row. The final stripped checkpoint stores
epoch -1, so the epoch selection is established by matching metrics and CSV.

Checkpoint SHA-256:
`5bca64f5f7baa7fcaa45a64ead6207014f63b718d26709400231869e2c145afd`.

| Metric | Best training validation row | Fresh best.pt validation |
|---|---:|---:|
| Precision | 96.620% | 96.828% |
| Recall | 94.953% | 94.916% |
| mAP@50 | 98.466% | 98.465% |
| mAP@50–95 | 75.977% | 76.015% |

Fresh validation used 200 images / 1,535 annotated regions on GPU. Results are
close to the recorded best row; the fresh evaluation loads the saved checkpoint
and reports its own metrics, rather than promising bit-for-bit reproduction.

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| open | 99.508% | 95.324% | 99.193% | 68.447% |
| short | 95.066% | 90.640% | 96.148% | 69.671% |
| mousebite | 98.046% | 94.813% | 98.465% | 73.564% |
| spur | 91.065% | 96.298% | 98.615% | 72.350% |
| copper | 98.666% | 95.652% | 99.277% | 85.830% |
| pin-hole | 98.617% | 96.769% | 99.091% | 86.229% |

Short has the lowest recall; spur has the lowest precision. Open/short box
localization merits review, given the gap between AP@50 and AP@50–95.
These are measured validation metrics, not a single accuracy percentage and
not equivalent to per-detection confidence.

The machine-readable review and plots are in
`outputs/reviews/deeppcb_validation_20261012_160752_611070/`.
The review script evaluated temporary validation copies; original datasets,
training results and checkpoint files were not overwritten. An earlier review
attempt produced plots in `outputs/reviews/deeppcb_validation/` but failed to save
JSON; the timestamped folder is the complete review.

The local Surface checkpoint exists and has the original six-class taxonomy,
but `runs/pcb_defect/baseline/results.csv` contains only one epoch (checkpoint
epoch 0). Its saved validation metrics are precision 81.166%, recall 71.935%,
mAP@50 80.426%, mAP@50–95 36.903%. This is a separate preliminary artifact, not
the completed DeepPCB detector or proof that the intended final surface model
has been supplied.

At the time of the 12 October validation review, the official 500-image DeepPCB
test split had not been evaluated. The frozen held-out test was completed on
14 October 2026; see [TEST_EVALUATION.md](TEST_EVALUATION.md). Validation source
groups are disjoint from training, but the official
test shares some groups with train/validation; it is not a board-disjoint test.
This optical dataset and detector do not establish CT capability or real-camera
performance. CT remains Not configured.
