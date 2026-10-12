"""Separate optical detector contracts; CT has no configured inference pipeline."""
from src.application.inspection import PCB_CLASSES, DEEPPCB_CLASSES

MODES = {
    "surface": {"label": "Surface", "classes": PCB_CLASSES,
                "checkpoint": "runs/pcb_defect/baseline/weights/best.pt",
                "scope": "PCB surface optical inspection; no CT or internal layer analysis."},
    "optical_bw": {"label": "Optical B/W", "classes": DEEPPCB_CLASSES,
                   "checkpoint": "runs/deeppcb/yolo11n_baseline/weights/best.pt",
                   "scope": "DeepPCB optical B/W defect detection on individual images; no template comparison or CT analysis."},
    "ct": {"label": "CT scans", "classes": frozenset(), "checkpoint": None,
           "scope": "Not configured. Requires a separate CT model and processing pipeline."},
}


def matches_mode(names, mode):
    return mode != "ct" and len(names) == 6 and frozenset(names) == MODES[mode]["classes"]


def mode_defaults(config, mode):
    settings = {**config.get("inference", {}), "imgsz": config.get("image", {}).get("width", 640),
                "checkpoint": MODES[mode]["checkpoint"]}
    if mode == "surface":
        settings["checkpoint"] = config.get("model", {}).get("best_checkpoint", settings["checkpoint"])
    settings.update(config.get("application", {}).get("modes", {}).get(mode, {}))
    return settings
