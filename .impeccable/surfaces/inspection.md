# Inspection workspace

Mode: Operate. Target: app/app.py, app/theme.py, app/studio.css.

## Direction contract

THESIS: Make PCB image inspection feel like a board workspace, with clear image
intake, review space, and separate inference controls.

OWN-WORLD: Solder-mask green surfaces, copper pads, silkscreen white text and
crisp routing geometry. Wix wh-1081 reference: oversized light Barlow headings,
wireframe layer artwork, square edges, fine dividers and broad action bands.
Solid colours follow an approximate 60–30–10 hierarchy: deep green
background, lighter green surfaces, restrained copper and signal-green accents.
Controls retain Streamlit semantics.

STORY: Select the mode and device, identify the board, choose an image, inspect,
review the original and marked regions, then download traceable reports or view
mode-specific history. Confidence is never presented as measured accuracy.

FIRST VIEWPORT: Large two-line “Multilayer PCB Defect Detector” title over PCB
wireframes, simple Workspace/Scope anchor navigation, broad copper entry link.
Below: horizontal mode/device/cutoff bar, advanced settings collapsed, and
three full-width workflow tabs. Dominant board canvas left and compact image
intake right, in a 1.85:1 split. Mobile stacks controls and puts intake before
the canvas. Results render inside the canvas, avoiding repeated source images.

FORM: Direct implementation of the user's explicit PCB architecture brief.
No seed key: Impeccable engine/context launcher could not run because its
installation needed unavailable cache write permissions; manual inspection
and independent review substitute for engine checks in this session.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Signature interaction: circuit signal connection, 380ms eased tab-panel entrance
and 240ms hover/focus/pressed transitions. Reduced motion disables authored
animations and transitions. Artwork is native vector geometry, with
no generated raster assets.

Verification: existing application/DeepPCB regressions; desktop and mobile
captures under outputs/reviews/frontend; independent finish review.
