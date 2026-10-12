"""PCB inspection modes. Native optical inference runs outside Streamlit."""
import faulthandler
import hashlib
import logging
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.application.inspection import (
    decode_image, inspection_key, resolve_checkpoint, run_worker,
)
from src.application.modes import MODES, matches_mode, mode_defaults
from src.application.reports import build_report, render_overlay, report_payload
from src.application.history import save_scan, statistics
from src.inference.predict import filter_detections
from src.utils.config import load_config

st.set_page_config(page_title="PCB Inspection Studio", page_icon="🔬", layout="wide")
LOGGER = logging.getLogger("pcb.app")


@st.cache_resource
def enable_diagnostics():
    directory = ROOT / "outputs" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "app-native.log").open("a", encoding="utf-8")
    faulthandler.enable(file=handle)
    if not LOGGER.handlers:
        logger_handle = logging.FileHandler(directory / "app.log", encoding="utf-8")
        logger_handle.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(logger_handle)
        LOGGER.setLevel(logging.INFO)
    return handle


@st.cache_data(show_spinner=False, max_entries=8)
def checkpoint_metadata(path, size, modified):
    return run_worker(Path(path))


def new_snapshot(image, raw, data, checkpoint, filename, pcb_id, device, threshold, mode="surface"):
    stat = checkpoint.stat()
    result = filter_detections(raw, threshold)
    metadata = {"scan_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                "pcb_id": pcb_id.strip() or "Unspecified", "filename": filename,
                "image_sha256": hashlib.sha256(data).hexdigest(), "dimensions": list(image.size),
                "model_path": str(checkpoint), "device": device, "mode": mode,
                "checkpoint_identity": f"{checkpoint.name}:{stat.st_size}:{stat.st_mtime_ns}"}
    return {"result": result, "metadata": metadata, "image": image}


def show_result(snapshot, threshold, db_path):
    result = filter_detections(snapshot["result"], threshold)
    metadata = snapshot["metadata"]
    # One bounded rendered view; threshold changes replace bytes rather than accumulate.
    view_key = (metadata["scan_id"], threshold)
    if st.session_state.get("render_key") != view_key:
        overlay = render_overlay(snapshot["image"], result)
        report_html, report_json = build_report(result, metadata, overlay)
        st.session_state["render_key"] = view_key
        st.session_state["rendered"] = (overlay, report_html, report_json)
    overlay, report_html, report_json = st.session_state["rendered"]
    a, b, c, d = st.columns(4)
    a.metric("Result", "Review required" if result["defect_count"] else "No detections")
    b.metric("Defect regions", result["defect_count"])
    c.metric("Inference pipeline", f'{result["processing_time_ms"]:.0f} ms')
    d.metric("Confidence cutoff", f"{threshold:.0%}")
    left, right = st.columns(2)
    left.image(snapshot["image"], caption=f'Original {MODES[metadata["mode"]]["label"]} image', width="stretch")
    right.image(overlay, caption="Detected regions at current cutoff", width="stretch")
    if result["detections"]:
        st.dataframe([{"Type": d["class_name"], "Confidence (%)": round(d["confidence"], 2),
                       "Bounding box (px)": str(d["bbox"])} for d in result["detections"]], hide_index=True)
    else:
        st.info("No regions exceeded this cutoff. This does not certify a defect-free board.")
    st.caption("Threshold changes preview the same scan; history keeps the cutoff selected when Inspect was pressed.")
    stem = f'pcb_{metadata["mode"]}_{metadata["scan_id"][:8]}_{int(threshold * 100)}'
    one, two, three = st.columns(3)
    one.download_button("Download printable report", report_html, file_name=stem + ".html",
                        mime="text/html", on_click="ignore", key="report_download")
    two.download_button("Download JSON", report_json, file_name=stem + ".json",
                        mime="application/json", on_click="ignore", key="json_download")
    three.download_button("Download annotated PNG", overlay, file_name=stem + ".png",
                          mime="image/png", on_click="ignore", key="png_download")
    st.caption("Open the HTML report in a browser and use Print → Save as PDF. Reports reflect the current preview cutoff.")
    if not snapshot.get("saved"):
        try:
            save_scan(db_path, report_payload(snapshot["result"], metadata))
            snapshot["saved"] = True
        except Exception:
            LOGGER.exception("Could not save inspection history")
            st.warning("Result is available, but history could not be saved. Check outputs/logs/app.log and folder permissions.")


@st.fragment(run_every="10s")
def show_history(db_path, mode):
    st.subheader("Inspection activity")
    period = st.selectbox("Time period", [24, 168, 720], format_func=lambda h: {24: "Last 24 hours", 168: "Last 7 days", 720: "Last 30 days"}[h])
    try:
        stats = statistics(db_path, period, mode=mode)
    except Exception:
        LOGGER.exception("Could not read scan history")
        st.warning("History is unavailable. Check outputs/logs/app.log.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Scans", stats["scans"])
    b.metric("Flagged scans", stats["flagged"])
    c.metric("Defect regions", stats["defects"])
    d.metric("Mean detection confidence", "—" if stats["mean_confidence"] is None else f'{stats["mean_confidence"]:.1f}%')
    st.caption("Refreshes every 10 seconds. Counts represent explicit inspections, including deliberate rescans. Confidence is not accuracy.")
    st.caption("History is filtered by inspection mode. Older scans without a recorded mode remain in SQLite and are excluded from these totals.")
    if stats["records"]:
        left, right = st.columns(2)
        left.bar_chart(stats["timeline"], x_label="Hour (UTC)", y_label="Scans")
        right.bar_chart(stats["types"], x_label="Defect type", y_label="Regions")
        st.dataframe([{"UTC time": r["created_at"], "PCB": r["pcb_id"], "File": r["filename"],
                       "Defects": r["defect_count"], "Cutoff": r["threshold"], "Scan ID": r["scan_id"]}
                      for r in stats["records"]], hide_index=True)
    else:
        st.info("Run an inspection to start your scan history.")
    st.write("**Accuracy score: unavailable.** Labeled test images are required to measure model performance.")


def main():
    enable_diagnostics()
    st.title("PCB Inspection Studio")
    st.caption("Surface • Optical B/W • CT scans • traceable scan reports")
    config = load_config(str(ROOT / "config.yaml"))
    db_path = ROOT / config.get("application", {}).get("database", "outputs/inspection.sqlite3")
    st.sidebar.header("Inspection settings")
    mode = st.sidebar.selectbox("Inspection mode", list(MODES), index=1,
                                format_func=lambda value: MODES[value]["label"], key="inspection_mode")
    st.caption(MODES[mode]["scope"])
    if mode == "ct":
        st.info("CT scans — Not configured")
        st.write("CT requires its own model, volume/slice processing and labeled validation. PCB-CTSeg is a proposed starting point for pads, traces and vias segmentation; these structural labels do not establish manufacturing fault detection.")
        return
    # Non-widget state survives Streamlit's cleanup when another mode is selected.
    settings = st.session_state.setdefault("mode_settings", {})
    inference = settings.setdefault(mode, mode_defaults(config, mode))
    selected = st.sidebar.text_input("Trained checkpoint", value=inference["checkpoint"], key=f"checkpoint_{mode}")
    device = st.sidebar.selectbox("Inference device", ["cpu", "0"],
                                  index=1 if inference.get("device") == "0" else 0,
                                  format_func=lambda d: "CPU (stable baseline)" if d == "cpu" else "CUDA GPU 0", key=f"device_{mode}")
    threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95,
                                  float(inference.get("confidence_threshold", 0.25)), 0.05, key=f"threshold_{mode}")
    imgsz = st.sidebar.selectbox("Inference image size", [320, 640, 960, 1280],
                                index=[320, 640, 960, 1280].index(inference.get("imgsz", 640)), key=f"imgsz_{mode}")
    iou = st.sidebar.slider("IoU threshold", 0.1, 0.9, float(inference.get("iou_threshold", 0.45)), 0.05, key=f"iou_{mode}")
    inference.update(checkpoint=selected, device=device, confidence_threshold=threshold, imgsz=imgsz, iou_threshold=iou)
    st.sidebar.caption("Adjusting the cutoff filters cached candidates; it does not run the model again.")
    st.sidebar.subheader("Supported defects")
    st.sidebar.caption(", ".join(sorted(MODES[mode]["classes"])))
    inspect, history, setup = st.tabs(["Inspect a PCB", "Scan history", "Scope & presentation setup"])
    with inspect:
        st.subheader("Upload and inspect")
        st.write("Give the PCB an identifier, choose an image, then run an inspection.")
        pcb_id = st.text_input("PCB / batch identifier", max_chars=100)
        uploaded = st.file_uploader(f'{MODES[mode]["label"]} image', type=["png", "jpg", "jpeg"], key=f"upload_{mode}")
        sample_dir = ROOT / "data/deeppcb_yolo/images/val" if mode == "optical_bw" else ROOT / config["data"]["raw_dir"] / "val" / "images"
        samples = sorted(sample_dir.glob("*.jpg"))[:20]
        sample = st.selectbox("Or use a local validation sample", [None, *samples], format_func=lambda p: p.name if p else "Choose a sample", key=f"sample_{mode}") if samples else None
        data, filename = None, None
        if uploaded is not None:
            data, filename = uploaded.getvalue(), uploaded.name
        elif sample:
            data, filename = sample.read_bytes(), sample.name
        try:
            checkpoint = resolve_checkpoint(selected)
            stat = checkpoint.stat()
            with st.spinner("Checking checkpoint..."):
                info = checkpoint_metadata(str(checkpoint), stat.st_size, stat.st_mtime_ns)
            if not matches_mode(info["names"], mode):
                st.error(f'This checkpoint does not match the {MODES[mode]["label"]} taxonomy. Select its PCB-trained best.pt. Generic yolo11n.pt / yolo26n.pt weights cannot inspect PCB defects.')
                checkpoint = None
            else:
                st.sidebar.success(f"PCB model ready: {checkpoint.name}")
                st.sidebar.write(", ".join(info["names"]).replace("_", " "))
        except Exception as exc:
            LOGGER.exception("Checkpoint unavailable")
            st.error(str(exc))
            checkpoint = None
        image, key = None, None
        if data is not None:
            try:
                image = decode_image(data)
                if checkpoint:
                    key = mode + ":" + inspection_key(data, checkpoint, device, imgsz, iou)
            except Exception as exc:
                LOGGER.exception("Image rejected")
                st.error(f"Cannot read this image: {exc}")
        scan = st.button("Inspect PCB", type="primary", disabled=image is None or checkpoint is None)
        if scan:
            try:
                if st.session_state.get("raw_key") != key:
                    with st.spinner("Inspecting in an isolated worker..."):
                        output = run_worker(checkpoint, image, device=device, imgsz=imgsz, iou=iou)
                    if not matches_mode(output["names"], mode):
                        raise ValueError("Checkpoint changed to a model incompatible with this inspection mode.")
                    st.session_state["raw_prediction"] = output["prediction"]
                    st.session_state["raw_key"] = key
                st.session_state["snapshot"] = new_snapshot(image, st.session_state["raw_prediction"], data,
                                                             checkpoint, filename, pcb_id, device, threshold, mode)
                st.session_state["snapshot_key"] = key
            except Exception as exc:
                st.session_state.pop("snapshot", None)
                LOGGER.exception("Inspection failed")
                st.error(str(exc))
        if key is not None and st.session_state.get("snapshot_key") == key and "snapshot" in st.session_state:
            show_result(st.session_state["snapshot"], threshold, db_path)
        elif image is not None:
            st.image(image, caption=f"Ready to inspect: {filename}", width="stretch")
    with history:
        show_history(db_path, mode)
    with setup:
        st.subheader("What this prototype does")
        st.write(MODES[mode]["scope"])
        st.write("Surface uses the original surface-trained checkpoint. Optical B/W uses a separate DeepPCB checkpoint with open, short, mousebite, spur, copper and pin-hole classes. Pin-hole is distinct from missing_hole.")
        st.write("Confidence describes individual predictions. Measured precision, recall and mAP require labeled evaluation linked to the selected checkpoint and split.")
        st.subheader("Future optical hardware setup — presentation only")
        st.markdown("1. PCB held flat in a repeatable positioning jig inside a matte enclosure.\n2. Overhead USB camera with a close-focus lens on a fixed stand.\n3. Diffuse ring light powered from a suitable supply; avoid reflections.\n4. Camera USB → laptop running this app. Save a PNG/JPEG and use the same upload pipeline.\n5. Validate on labeled images from the actual camera before reporting hardware accuracy.")
        st.write("DeepPCB supplies paired optical images, not CT data. A CT model will require an appropriate CT dataset and imaging source; it is a later phase.")
        st.caption("Camera integration and CT processing are future work. CT scans remain Not configured.")


if __name__ == "__main__":
    main()
