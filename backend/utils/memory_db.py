"""
backend/utils/memory_db.py

MedGuard AI — Persistent Database Memory Layer
===============================================
Pure Python stdlib (sqlite3 + json) — works with ZERO external dependencies.
Stores and retrieves:
  • Detection events  (with full metadata, snapshots)
  • Patient profiles  (conditions, medications, contacts)
  • Chat history      (per-session, per-patient)
  • Agent actions     (every emergency response step)
  • System snapshots  (camera status, model metrics)
  • Alert logs        (channel, recipient, outcome)

This module is a SELF-CONTAINED drop-in that works even when
PostgreSQL, Redis or Kafka are not running (dev / edge mode).
"""

import sqlite3
import json
import hashlib
import uuid
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path


# ── Config ────────────────────────────────────────────────────
DB_DIR = Path(os.environ.get("MEDGUARD_DB_DIR", "./data"))
DB_PATH = DB_DIR / "medguard_memory.db"
SCHEMA_VERSION = 3


# ── Connection pool (simple) ──────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def _db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────

SCHEMA_SQL = """
-- ── Schema version ──────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── Patients ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    age               INTEGER,
    gender            TEXT,
    blood_group       TEXT,
    conditions        TEXT DEFAULT '[]',   -- JSON array
    medications       TEXT DEFAULT '[]',
    allergies         TEXT DEFAULT '[]',
    emergency_contacts TEXT DEFAULT '[]',  -- [{name,phone,relation}]
    camera_ids        TEXT DEFAULT '[]',
    notes             TEXT DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- ── Detection Events ─────────────────────────────────
CREATE TABLE IF NOT EXISTS detection_events (
    id              TEXT PRIMARY KEY,
    patient_id      TEXT REFERENCES patients(id),
    camera_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,   -- fall|seizure|cardiac|unconscious|facial_distress
    severity        TEXT NOT NULL,   -- low|medium|high|critical
    confidence      REAL NOT NULL,
    metadata        TEXT DEFAULT '{}',
    snapshot_path   TEXT,
    video_clip_path TEXT,
    latitude        REAL,
    longitude       REAL,
    alert_sent      INTEGER DEFAULT 0,
    resolved        INTEGER DEFAULT 0,
    resolved_by     TEXT,
    resolved_at     TEXT,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_det_camera   ON detection_events(camera_id);
CREATE INDEX IF NOT EXISTS idx_det_type     ON detection_events(event_type);
CREATE INDEX IF NOT EXISTS idx_det_ts       ON detection_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_det_severity ON detection_events(severity);

-- ── Alert Logs ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alert_logs (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL REFERENCES detection_events(id),
    channel     TEXT NOT NULL,   -- sms|email|call|push|webhook
    recipient   TEXT NOT NULL,
    status      TEXT NOT NULL,   -- sent|failed|pending|simulated
    response    TEXT,
    sent_at     TEXT NOT NULL
);

-- ── Chat History ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    patient_id  TEXT REFERENCES patients(id),
    role        TEXT NOT NULL,   -- user|assistant|system
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_session  ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_patient  ON chat_messages(patient_id);

-- ── Agent Memory ──────────────────────────────────────
-- Every tool call + decision the emergency agent makes
CREATE TABLE IF NOT EXISTS agent_memory (
    id          TEXT PRIMARY KEY,
    event_id    TEXT REFERENCES detection_events(id),
    agent_run   TEXT NOT NULL,   -- UUID grouping one full agent invocation
    step        INTEGER NOT NULL,
    action_type TEXT NOT NULL,   -- tool_call|decision|reasoning
    tool_name   TEXT,
    tool_input  TEXT DEFAULT '{}',
    tool_output TEXT,
    reasoning   TEXT,
    timestamp   TEXT NOT NULL
);

-- ── System Snapshots ─────────────────────────────────
-- Periodic health + metrics persistence
CREATE TABLE IF NOT EXISTS system_snapshots (
    id          TEXT PRIMARY KEY,
    cameras     TEXT DEFAULT '{}',   -- {cam_id: {fps, errors, running}}
    model_stats TEXT DEFAULT '{}',   -- {model: {latency_ms, calls}}
    ws_clients  INTEGER DEFAULT 0,
    detections_hour INTEGER DEFAULT 0,
    alerts_hour     INTEGER DEFAULT 0,
    cpu_pct     REAL,
    mem_pct     REAL,
    captured_at TEXT NOT NULL
);

-- ── Medical Documents (for RAG) ───────────────────────
CREATE TABLE IF NOT EXISTS medical_documents (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT REFERENCES patients(id),
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    doc_type    TEXT DEFAULT 'general', -- report|prescription|history|lab
    source      TEXT,
    checksum    TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_patient ON medical_documents(patient_id);

-- ── Notification Preferences ─────────────────────────
CREATE TABLE IF NOT EXISTS notification_prefs (
    id              TEXT PRIMARY KEY,
    patient_id      TEXT REFERENCES patients(id),
    channel         TEXT NOT NULL,   -- sms|email|push|call
    enabled         INTEGER DEFAULT 1,
    threshold       TEXT DEFAULT 'high',  -- low|medium|high|critical
    address         TEXT NOT NULL,   -- phone/email/token
    updated_at      TEXT NOT NULL
);
"""


# ── Initialise ────────────────────────────────────────────────

def init_memory_db() -> str:
    """Create all tables and return DB path."""
    with _db() as conn:
        conn.executescript(SCHEMA_SQL)
        # Check / update schema version
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta VALUES ('version', ?)",
            (str(SCHEMA_VERSION),)
        )
    return str(DB_PATH)


# ═══════════════════════════════════════════════════════════════
#  PATIENT MEMORY
# ═══════════════════════════════════════════════════════════════

def save_patient(
    name: str,
    age: int = 0,
    gender: str = "",
    blood_group: str = "",
    conditions: List[str] = None,
    medications: List[str] = None,
    allergies: List[str] = None,
    emergency_contacts: List[Dict] = None,
    camera_ids: List[str] = None,
    notes: str = "",
    patient_id: str = None,
) -> str:
    """Create or update a patient. Returns patient_id."""
    pid = patient_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM patients WHERE id=?", (pid,)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE patients SET
                    name=?, age=?, gender=?, blood_group=?,
                    conditions=?, medications=?, allergies=?,
                    emergency_contacts=?, camera_ids=?, notes=?, updated_at=?
                WHERE id=?
            """, (
                name, age, gender, blood_group,
                json.dumps(conditions or []),
                json.dumps(medications or []),
                json.dumps(allergies or []),
                json.dumps(emergency_contacts or []),
                json.dumps(camera_ids or []),
                notes, now, pid
            ))
        else:
            conn.execute("""
                INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pid, name, age, gender, blood_group,
                json.dumps(conditions or []),
                json.dumps(medications or []),
                json.dumps(allergies or []),
                json.dumps(emergency_contacts or []),
                json.dumps(camera_ids or []),
                notes, now, now
            ))

    return pid


def get_patient(patient_id: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)
        ).fetchone()
    return _patient_row(row) if row else None


def list_patients() -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY name"
        ).fetchall()
    return [_patient_row(r) for r in rows]


def _patient_row(row) -> Dict:
    d = dict(row)
    for field in ("conditions", "medications", "allergies",
                  "emergency_contacts", "camera_ids"):
        try:
            d[field] = json.loads(d[field])
        except Exception:
            d[field] = []
    return d


# ═══════════════════════════════════════════════════════════════
#  DETECTION EVENT MEMORY
# ═══════════════════════════════════════════════════════════════

def save_detection_event(
    camera_id: str,
    event_type: str,
    severity: str,
    confidence: float,
    metadata: Dict = None,
    patient_id: str = None,
    snapshot_path: str = None,
    video_clip_path: str = None,
    latitude: float = None,
    longitude: float = None,
    event_id: str = None,
) -> str:
    """Persist a detection event. Returns event_id."""
    eid = event_id or str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    with _db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO detection_events
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            eid, patient_id, camera_id, event_type, severity,
            round(confidence, 4),
            json.dumps(metadata or {}),
            snapshot_path, video_clip_path,
            latitude, longitude,
            0, 0, None, None, ts
        ))

    return eid


def get_detection_events(
    limit: int = 100,
    camera_id: str = None,
    event_type: str = None,
    severity: str = None,
    resolved: bool = None,
    since_hours: int = None,
    patient_id: str = None,
) -> List[Dict]:
    """Query detection events with filters."""
    q = "SELECT * FROM detection_events WHERE 1=1"
    params = []

    if camera_id:
        q += " AND camera_id=?"; params.append(camera_id)
    if event_type:
        q += " AND event_type=?"; params.append(event_type)
    if severity:
        q += " AND severity=?"; params.append(severity)
    if resolved is not None:
        q += " AND resolved=?"; params.append(1 if resolved else 0)
    if patient_id:
        q += " AND patient_id=?"; params.append(patient_id)
    if since_hours:
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        q += " AND timestamp >= ?"; params.append(cutoff)

    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _db() as conn:
        rows = conn.execute(q, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
        d["alert_sent"] = bool(d["alert_sent"])
        d["resolved"] = bool(d["resolved"])
        result.append(d)

    return result


def resolve_event(event_id: str, resolved_by: str = "user") -> bool:
    with _db() as conn:
        conn.execute(
            "UPDATE detection_events SET resolved=1, resolved_by=?, resolved_at=? WHERE id=?",
            (resolved_by, datetime.utcnow().isoformat(), event_id)
        )
    return True


def mark_alert_sent(event_id: str) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE detection_events SET alert_sent=1 WHERE id=?", (event_id,)
        )


def get_event_stats(hours: int = 24) -> Dict:
    """Summary statistics for the last N hours."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with _db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM detection_events WHERE timestamp>=?", (cutoff,)
        ).fetchone()[0]

        by_type = conn.execute(
            "SELECT event_type, COUNT(*) c FROM detection_events "
            "WHERE timestamp>=? GROUP BY event_type ORDER BY c DESC",
            (cutoff,)
        ).fetchall()

        by_severity = conn.execute(
            "SELECT severity, COUNT(*) c FROM detection_events "
            "WHERE timestamp>=? GROUP BY severity",
            (cutoff,)
        ).fetchall()

        unresolved = conn.execute(
            "SELECT COUNT(*) FROM detection_events WHERE resolved=0 AND timestamp>=?",
            (cutoff,)
        ).fetchone()[0]

        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM detection_events WHERE timestamp>=?",
            (cutoff,)
        ).fetchone()[0]

    return {
        "period_hours": hours,
        "total_events": total,
        "unresolved": unresolved,
        "avg_confidence": round(avg_conf or 0, 3),
        "by_type": {r["event_type"]: r["c"] for r in by_type},
        "by_severity": {r["severity"]: r["c"] for r in by_severity},
    }


# ═══════════════════════════════════════════════════════════════
#  ALERT LOG MEMORY
# ═══════════════════════════════════════════════════════════════

def save_alert_log(
    event_id: str,
    channel: str,
    recipient: str,
    status: str,
    response: str = "",
) -> str:
    lid = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO alert_logs VALUES (?,?,?,?,?,?,?)",
            (lid, event_id, channel, recipient, status,
             response, datetime.utcnow().isoformat())
        )
    return lid


def get_alert_logs(event_id: str = None, limit: int = 50) -> List[Dict]:
    with _db() as conn:
        if event_id:
            rows = conn.execute(
                "SELECT * FROM alert_logs WHERE event_id=? ORDER BY sent_at DESC LIMIT ?",
                (event_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alert_logs ORDER BY sent_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
#  CHAT HISTORY MEMORY
# ═══════════════════════════════════════════════════════════════

def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    patient_id: str = None,
    metadata: Dict = None,
) -> str:
    mid = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO chat_messages VALUES (?,?,?,?,?,?,?)",
            (mid, session_id, patient_id, role, content,
             json.dumps(metadata or {}),
             datetime.utcnow().isoformat())
        )
    return mid


def get_chat_history(
    session_id: str,
    limit: int = 50,
    patient_id: str = None,
) -> List[Dict]:
    with _db() as conn:
        if patient_id:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id=? AND patient_id=? "
                "ORDER BY timestamp ASC LIMIT ?",
                (session_id, patient_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id=? "
                "ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


def get_all_sessions(patient_id: str = None) -> List[Dict]:
    with _db() as conn:
        if patient_id:
            rows = conn.execute(
                "SELECT session_id, COUNT(*) msg_count, MIN(timestamp) started, "
                "MAX(timestamp) last_active "
                "FROM chat_messages WHERE patient_id=? "
                "GROUP BY session_id ORDER BY last_active DESC",
                (patient_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, COUNT(*) msg_count, MIN(timestamp) started, "
                "MAX(timestamp) last_active "
                "FROM chat_messages "
                "GROUP BY session_id ORDER BY last_active DESC LIMIT 50"
            ).fetchall()
    return [dict(r) for r in rows]


def clear_session(session_id: str) -> int:
    with _db() as conn:
        cursor = conn.execute(
            "DELETE FROM chat_messages WHERE session_id=?", (session_id,)
        )
    return cursor.rowcount


# ═══════════════════════════════════════════════════════════════
#  AGENT MEMORY
# ═══════════════════════════════════════════════════════════════

def save_agent_step(
    event_id: str,
    agent_run: str,
    step: int,
    action_type: str,
    tool_name: str = None,
    tool_input: Dict = None,
    tool_output: str = None,
    reasoning: str = None,
) -> str:
    sid = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO agent_memory VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, event_id, agent_run, step, action_type,
             tool_name,
             json.dumps(tool_input or {}),
             tool_output, reasoning,
             datetime.utcnow().isoformat())
        )
    return sid


def get_agent_history(event_id: str) -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_memory WHERE event_id=? ORDER BY step ASC",
            (event_id,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["tool_input"] = json.loads(d["tool_input"])
        except Exception:
            d["tool_input"] = {}
        result.append(d)
    return result


# ═══════════════════════════════════════════════════════════════
#  SYSTEM SNAPSHOT MEMORY
# ═══════════════════════════════════════════════════════════════

def save_system_snapshot(
    cameras: Dict = None,
    model_stats: Dict = None,
    ws_clients: int = 0,
    detections_hour: int = 0,
    alerts_hour: int = 0,
    cpu_pct: float = 0.0,
    mem_pct: float = 0.0,
) -> str:
    sid = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO system_snapshots VALUES (?,?,?,?,?,?,?,?,?)",
            (sid,
             json.dumps(cameras or {}),
             json.dumps(model_stats or {}),
             ws_clients, detections_hour, alerts_hour,
             cpu_pct, mem_pct,
             datetime.utcnow().isoformat())
        )
    return sid


def get_system_history(hours: int = 6) -> List[Dict]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM system_snapshots WHERE captured_at>=? ORDER BY captured_at DESC",
            (cutoff,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        for f in ("cameras", "model_stats"):
            try:
                d[f] = json.loads(d[f])
            except Exception:
                d[f] = {}
        result.append(d)
    return result


# ═══════════════════════════════════════════════════════════════
#  MEDICAL DOCUMENTS (for RAG ingestion tracking)
# ═══════════════════════════════════════════════════════════════

def save_medical_document(
    patient_id: str,
    title: str,
    content: str,
    doc_type: str = "general",
    source: str = "",
) -> str:
    did = str(uuid.uuid4())
    checksum = hashlib.md5(content.encode()).hexdigest()

    with _db() as conn:
        # Avoid duplicate ingestion
        existing = conn.execute(
            "SELECT id FROM medical_documents WHERE checksum=? AND patient_id=?",
            (checksum, patient_id)
        ).fetchone()
        if existing:
            return existing["id"]

        conn.execute(
            "INSERT INTO medical_documents VALUES (?,?,?,?,?,?,?,?)",
            (did, patient_id, title, content, doc_type, source,
             checksum, datetime.utcnow().isoformat())
        )
    return did


def get_patient_documents(patient_id: str) -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, patient_id, title, doc_type, source, ingested_at "
            "FROM medical_documents WHERE patient_id=? ORDER BY ingested_at DESC",
            (patient_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
#  NOTIFICATION PREFERENCES
# ═══════════════════════════════════════════════════════════════

def save_notification_pref(
    patient_id: str,
    channel: str,
    address: str,
    threshold: str = "high",
    enabled: bool = True,
) -> str:
    nid = str(uuid.uuid4())
    with _db() as conn:
        # Replace existing same patient+channel
        conn.execute(
            "DELETE FROM notification_prefs WHERE patient_id=? AND channel=?",
            (patient_id, channel)
        )
        conn.execute(
            "INSERT INTO notification_prefs VALUES (?,?,?,?,?,?,?)",
            (nid, patient_id, channel, int(enabled), threshold,
             address, datetime.utcnow().isoformat())
        )
    return nid


def get_notification_prefs(patient_id: str) -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM notification_prefs WHERE patient_id=? AND enabled=1",
            (patient_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════
#  FULL MEMORY EXPORT
# ═══════════════════════════════════════════════════════════════

def export_memory(output_path: str = None) -> Dict:
    """Export entire database to a JSON dict (for backup/debug)."""
    output_path = output_path or str(DB_DIR / f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")

    data = {
        "exported_at": datetime.utcnow().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "patients": list_patients(),
        "detection_events": get_detection_events(limit=10000),
        "alert_logs": get_alert_logs(limit=10000),
        "system_stats": get_event_stats(hours=168),  # 1 week
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return {"path": output_path, "patients": len(data["patients"]),
            "events": len(data["detection_events"])}


def get_db_info() -> Dict:
    """Return DB size, table row counts."""
    db_size = os.path.getsize(DB_PATH) if DB_PATH.exists() else 0
    tables = ["patients", "detection_events", "alert_logs",
              "chat_messages", "agent_memory", "system_snapshots",
              "medical_documents", "notification_prefs"]
    counts = {}
    with _db() as conn:
        for t in tables:
            try:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                counts[t] = 0

    return {
        "db_path": str(DB_PATH),
        "db_size_kb": round(db_size / 1024, 1),
        "row_counts": counts,
        "schema_version": SCHEMA_VERSION,
    }
