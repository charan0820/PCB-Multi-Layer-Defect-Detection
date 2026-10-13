# Multilayer PCB Defect Detector

<!-- impeccable:product-schema 1 -->

## Platform

web

## Product Purpose

Inspect individual PCB images with mode-specific trained detectors, review marked
regions, export traceable reports, and review scan history. This is a prototype;
detection confidence is not measured accuracy or board certification.

## Capabilities and Constraints

- Existing stack: Python and Streamlit, with isolated inference subprocesses.
- Surface and Optical B/W each retain independent model and inference settings.
- PNG/JPEG uploads and local validation samples; 15 MB upload limit.
- Confidence changes filter cached predictions without rerunning inference.
- Explicit inspections persist metadata and detections to SQLite.
- Exports: annotated PNG, JSON, and printable standalone HTML.
- CT processing and camera integration are not implemented.

## Brand Commitments

The user requested a PCB architecture-inspired frontend with gradients, resized
boxes, and smoother transitions, using Taste and Impeccable together.
The user subsequently requested a new layout, new fonts, added transitions, and
the exact product title "Multilayer PCB Defect Detector". The title does not
change the implemented optical scope; CT/internal-layer inference remains
unconfigured.
The latest visual direction replaces gradients with solid PCB colours and uses
the 60–30–10 rule for dominant background, supporting surfaces, and accents.
The user supplied Wix template wh-1081 as a visual reference. Its large light
headings, wireframe hero, square edges and action bands guide composition;
financial content and claims from the reference do not apply to this detector.

## Evidence on Hand

Product scope is documented in README.md and src/application/modes.py.
Regression coverage is in tests/test_application.py and tests/test_deeppcb.py.
No new accuracy claims or performance benchmarks are supplied by this redesign.
