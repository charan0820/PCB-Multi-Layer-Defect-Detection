# Presentation setup: PCB surface inspection

## Accurate project description

The current application analyses uploaded optical surface images of PCBs. It finds six defect types, draws their bounding boxes, generates an inspection report and records scan metadata/detections in SQLite. It is an academic inspection assistant; a no-detection result is not a quality certificate.

The source dataset primarily represents bare-board copper patterns. Performance on populated boards, different trace sizes, new illumination or new camera optics must be evaluated separately. The app does not inspect hidden copper layers or CT scans.

## Current demonstration

1. Install dependencies and supply the PCB-trained `best.pt` (not generic YOLO weights).
2. Launch the dashboard; use CPU first while confirming stability.
3. Enter PCB/batch ID, upload a surface image and press Inspect.
4. Explain the detected type, box coordinates and confidence. Change the cutoff to show how displayed regions change without another model run.
5. Download the printable HTML report and annotated image; show scan history and defect-type statistics.
6. Describe confidence as the model's prediction score. Actual performance must be measured against labeled ground truth using precision/recall/mAP.

## Planned optical test rig (not implemented)

| Component | Placement / role | Connection |
|---|---|---|
| USB camera with suitable close-focus lens | Mounted vertically; fixed focus and magnification; captures the entire test region | USB cable → laptop |
| Rigid camera stand | Keeps working distance and alignment repeatable | Mechanically fixed to enclosure/frame |
| Diffuse LED ring light | Around lens; uniform lighting without glare | Rated regulated DC supply → light |
| Matte enclosure | Blocks changing ambient light and reflections | Camera/light mounted inside; USB/power routed out |
| PCB positioning jig | Holds the bare PCB flat with repeatable orientation | Mechanically fixed to enclosure base |
| Laptop running the application | Executes YOLO inference, dashboard, SQLite and reports | Camera via USB; existing local application |

Choose the lens and field of view so the smallest relevant defect occupies enough pixels to be visible. Start with one fixed-size PCB/inspection region. Lock camera focus, exposure and white balance where supported; verify the traces are sharp, not saturated and not reflecting the light. Store a representative set of images and label them before evaluating hardware accuracy. Use an appropriately rated supply for the selected light.

The initial connection is simple: **camera → USB → laptop; regulated supply → ring light**. Captured JPEG/PNG images can already enter the upload flow manually. Automatic acquisition is a later phase, not part of this repair. If a large PCB needs multiple views, evaluate a fixed crop/tiling approach later rather than assuming the current whole-image detector preserves tiny defects after resizing.

## Second model boundary

DeepPCB provides aligned optical template/test image pairs captured with a linear-scan CCD. It is not a CT dataset. Its pin-hole class also needs deliberate mapping/evaluation relative to the current missing-hole class. A second optical model trained on DeepPCB does not automatically gain CT or internal-layer inspection capabilities. CT analysis requires appropriate CT training data and a separate imaging/validation plan.

Source: https://github.com/tangsanli5201/DeepPCB

Second-model work starts only after the current trained PCB model passes the acceptance sequence. Hardware integration remains the final stage.
