"""Thread-safe Pillow overlays and portable, printable inspection reports."""
from collections import Counter
import base64
import html
import io
import json

from PIL import ImageDraw, ImageFont

COLORS = ["#d32752", "#168441", "#9b7800", "#2559c5", "#bd590b", "#883eaf"]


def render_overlay(image, result):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, min(24, canvas.width // 55)))
    for detection in result["detections"]:
        box = detection.get("bbox", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(canvas.width - 1, x1)), max(0, min(canvas.width - 1, x2))))
        y1, y2 = sorted((max(0, min(canvas.height - 1, y1)), max(0, min(canvas.height - 1, y2))))
        color = COLORS[detection["class_id"] % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f'{detection["class_name"]} {detection["confidence"]:.1f}%'
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = bounds[2], bounds[3] + 4
        lx = max(0, min(x1, canvas.width - label_width - 6))
        ly = max(0, y1 - label_height - 4)
        draw.rectangle((lx, ly, lx + label_width + 6, ly + label_height + 2), fill=color)
        draw.text((lx + 3, ly), label, fill="white", font=font)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def report_payload(result, metadata):
    """Only retained detections are exported; never leak hidden raw candidates."""
    return {**metadata, **{key: value for key, value in result.items() if key != "raw_detections"},
            "defect_types": dict(Counter(d["class_name"] for d in result["detections"])),
            "scope": "Surface optical inspection only; no X-ray or internal layer inspection.",
            "accuracy": None,
            "accuracy_note": "Not measurable without labeled ground truth. Detection confidence is not accuracy."}


def build_report(result, metadata, overlay):
    payload = report_payload(result, metadata)
    escape = lambda value: html.escape(str(value), quote=True)
    fields = {"Scan ID": payload["scan_id"], "UTC timestamp": payload["created_at"],
              "PCB ID": payload["pcb_id"], "Source": payload["filename"],
              "Image size": payload["dimensions"], "Checkpoint": payload["model_path"],
              "Model identity": payload["checkpoint_identity"], "Device": payload["device"],
              "Status": payload["status"], "Defects": payload["defect_count"],
              "Confidence cutoff": f'{payload["threshold"] * 100:.0f}%',
              "Processing time": f'{payload["processing_time_ms"]:.1f} ms'}
    summary = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in fields.items())
    rows = "".join(f'<tr><td>{index}</td><td>{escape(d["class_name"])}</td><td>{d["confidence"]:.1f}%</td><td>{escape(d["bbox"])}</td></tr>'
                   for index, d in enumerate(result["detections"], 1))
    picture = base64.b64encode(overlay).decode("ascii")
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>PCB Inspection {escape(payload["scan_id"])}</title>
<style>body{{font:15px system-ui;color:#142b3c;max-width:960px;margin:40px auto;padding:0 24px}}
h1{{color:#087e78}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}
img{{max-width:100%}}.note{{background:#eef6f6;padding:16px}}@media print{{body{{margin:0}}img{{max-height:650px}}tr{{break-inside:avoid}}}}</style>
<h1>PCB Surface Inspection</h1><table>{summary}</table>
<img alt="Detected PCB regions" src="data:image/png;base64,{picture}">
<h2>Detected regions</h2><table><tr><th>#</th><th>Defect type</th><th>Confidence</th><th>Bounding box (pixels)</th></tr>{rows}</table>
<p class="note">No detected defects means none exceeded the selected cutoff; it does not certify a defect-free PCB.
Detection confidence is not an accuracy score. Accuracy requires labeled ground truth.<br>
Surface optical inspection only. No X-ray or internal layer analysis.</p></html>'''
    return document.encode("utf-8"), json.dumps(payload, indent=2).encode("utf-8")
