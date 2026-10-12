"""Transactional SQLite scan snapshots. Viewing/filtering never creates a scan."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute('''CREATE TABLE IF NOT EXISTS scans (
        scan_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, pcb_id TEXT NOT NULL,
        filename TEXT NOT NULL, payload TEXT NOT NULL)''')
    connection.execute("CREATE INDEX IF NOT EXISTS scans_time ON scans(created_at)")
    return connection


def save_scan(path, payload):
    connection = connect(path)
    try:
        with connection:
            connection.execute("INSERT OR IGNORE INTO scans VALUES (?, ?, ?, ?, ?)",
                (payload["scan_id"], payload["created_at"], payload["pcb_id"], payload["filename"], json.dumps(payload)))
    finally:
        connection.close()


def statistics(path, hours=24):
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    connection = connect(path)
    try:
        records = [json.loads(row[0]) for row in connection.execute(
            "SELECT payload FROM scans WHERE created_at >= ? ORDER BY created_at DESC", (since,))]
    finally:
        connection.close()
    counts, timeline = Counter(), Counter()
    for record in records:
        counts.update(d["class_name"] for d in record["detections"])
        timeline[record["created_at"][:13] + ":00 UTC"] += 1
    confidences = [d["confidence"] for r in records for d in r["detections"]]
    return {"scans": len(records), "flagged": sum(r["defect_count"] > 0 for r in records),
            "defects": sum(r["defect_count"] for r in records), "types": dict(counts),
            "timeline": dict(sorted(timeline.items())), "records": records,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None}
