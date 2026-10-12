"""Regression tests for upload identity, reruns, downloads and failure isolation."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.application.inspection import decode_image, inspection_key, run_worker, PCB_CLASSES
from src.application.reports import render_overlay, build_report, report_payload
from src.application.history import save_scan, statistics
from src.inference.predict import filter_detections

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (96, 64), color=color).save(stream, format="PNG")
    return stream.getvalue()


def raw_result():
    detections = [{"class_id": 0, "class_name": "mouse_bite", "confidence": 85., "confidence_score": .85, "bbox": [10, 10, 30, 30]},
                  {"class_id": 1, "class_name": "spur", "confidence": 40., "confidence_score": .40, "bbox": [40, 10, 50, 25]}]
    return {"detections": detections, "raw_detections": detections, "model_name": "fixture", "processing_time_ms": 12.5}


def metadata():
    return {"scan_id": "test-scan", "created_at": datetime.now(timezone.utc).isoformat(), "pcb_id": "PCB-1",
            "filename": '<script>alert(1)</script>.png', "dimensions": [96, 64], "model_path": "best.pt",
            "checkpoint_identity": "best:1:123", "device": "cpu"}


def test_image_validation_and_cache_identity(tmp_path):
    with pytest.raises(Exception):
        decode_image(b"broken png")
    assert decode_image(image_bytes()).mode == "RGB"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    key = inspection_key(image_bytes(), checkpoint, "cpu", 640, .45)
    assert key != inspection_key(image_bytes("blue"), checkpoint, "cpu", 640, .45)
    checkpoint.write_bytes(b"new checkpoint")
    assert key != inspection_key(image_bytes(), checkpoint, "cpu", 640, .45)
    assert key != inspection_key(image_bytes(), checkpoint, "0", 640, .45)


def test_reports_escape_input_and_keep_current_threshold():
    result = filter_detections(raw_result(), .5)
    overlay = render_overlay(decode_image(image_bytes()), result)
    report, data = build_report(result, metadata(), overlay)
    assert report.startswith(b"<!doctype html>")
    assert b"<script>" not in report
    assert b"&lt;script&gt;" in report
    payload = json.loads(data)
    assert payload["defect_count"] == 1
    assert "raw_detections" not in payload
    assert payload["accuracy"] is None
    assert Image.open(io.BytesIO(overlay)).size == (96, 64)


def test_history_is_idempotent_and_threshold_preview_does_not_change_snapshot(tmp_path):
    db = tmp_path / "scans.sqlite3"
    result = filter_detections(raw_result(), .25)
    payload = report_payload(result, metadata())
    save_scan(db, payload)
    save_scan(db, payload)
    filter_detections(result, .95)
    stats = statistics(db)
    assert stats["scans"] == 1
    assert stats["defects"] == 2
    assert stats["types"] == {"mouse_bite": 1, "spur": 1}


def test_worker_native_exit_and_timeout_leave_parent_alive():
    with patch("src.application.inspection.subprocess.run", return_value=subprocess.CompletedProcess([], -11)):
        with pytest.raises(RuntimeError, match="exit -11"):
            run_worker(Path("fake.pt"))
    with patch("src.application.inspection.subprocess.run", side_effect=subprocess.TimeoutExpired([], 1)):
        with pytest.raises(RuntimeError, match="timed out"):
            run_worker(Path("fake.pt"), timeout=1)


def test_app_threshold_downloads_and_history_do_not_rerun_inference(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    cfg = {"model": {"best_checkpoint": str(checkpoint)}, "image": {"width": 640},
           "data": {"raw_dir": "no-dataset"}, "inference": {"confidence_threshold": .25},
           "application": {"database": str(tmp_path / "ui.sqlite3")}}
    calls = []
    def worker(path, image=None, **kwargs):
        if image is not None:
            calls.append(1)
        return {"names": list(PCB_CLASSES), "prediction": raw_result()}
    upload = io.BytesIO(image_bytes())
    upload.name = 'pcb.png'
    monkeypatch.setattr("src.application.inspection.run_worker", worker)
    monkeypatch.setattr("src.utils.config.load_config", lambda path: cfg)
    with patch("streamlit.file_uploader", return_value=upload):
        app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=20).run()
        assert not app.exception
        app.button[0].click().run()
        assert not app.exception
        assert len(calls) == 1
        scan_id = app.session_state["snapshot"]["metadata"]["scan_id"]
        for threshold in (.5, .95, .05, .25):
            app.slider[0].set_value(threshold).run()
            assert not app.exception
            assert len(calls) == 1
            assert app.session_state["snapshot"]["metadata"]["scan_id"] == scan_id
        # Downloads are explicit no-rerun widgets with correct MIME and extension.
        downloads = app.get("download_button")
        assert len(downloads) == 3
        assert all(d.proto.ignore_rerun for d in downloads)
        assert statistics(tmp_path / "ui.sqlite3")["scans"] == 1
        # Re-inspection is explicit and creates a new snapshot, reusing raw pixels.
        app.button[0].click().run()
        assert not app.exception
        assert len(calls) == 1
        assert statistics(tmp_path / "ui.sqlite3")["scans"] == 2
