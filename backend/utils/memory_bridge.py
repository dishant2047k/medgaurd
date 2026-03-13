"""
backend/utils/memory_bridge.py

Bridges the real-time inference engine → database memory.
Called from api/main.py on_medical_event().
Also provides memory-aware helpers for chat and agent.
"""
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from backend.utils.memory_db import (
    save_detection_event,
    save_alert_log,
    save_agent_step,
    save_system_snapshot,
    save_chat_message,
    get_chat_history,
    get_event_stats,
    mark_alert_sent,
    get_patient,
    list_patients,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOT_DIR", "./data/snapshots"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def persist_medical_event(event) -> str:
    """
    Called by the inference engine callback.
    Saves snapshot image to disk, persists event to DB.
    Returns the event_id.
    """
    snapshot_path = None

    # Save snapshot image
    if event.snapshot_bytes:
        fname = f"{event.camera_id}_{event.event_type}_{int(event.timestamp)}.jpg"
        snap_path = SNAPSHOT_DIR / fname
        try:
            snap_path.write_bytes(event.snapshot_bytes)
            snapshot_path = str(snap_path)
        except Exception as e:
            logger.warning("snapshot_save_failed", error=str(e))

    # Look up patient by camera ID
    patient_id = _find_patient_by_camera(event.camera_id)

    event_id = save_detection_event(
        camera_id=event.camera_id,
        event_type=event.event_type,
        severity=event.severity,
        confidence=event.confidence,
        metadata=event.metadata,
        patient_id=patient_id,
        snapshot_path=snapshot_path,
        latitude=float(os.environ.get("DEFAULT_LATITUDE", "28.6139")),
        longitude=float(os.environ.get("DEFAULT_LONGITUDE", "77.2090")),
    )

    logger.info("event_persisted",
                event_id=event_id,
                event_type=event.event_type,
                severity=event.severity,
                patient_id=patient_id)

    return event_id


def persist_alert_result(event_id: str, channel: str,
                          recipient: str, status: str, response: str = ""):
    """Save the outcome of an alert dispatch."""
    save_alert_log(event_id, channel, recipient, status, response)
    if status in ("sent", "simulated"):
        mark_alert_sent(event_id)


def persist_agent_steps(event_id: str, agent_run: str, messages: list):
    """
    Parse LangGraph message list and save each tool call / response.
    """
    step = 0
    for msg in messages:
        try:
            # Tool call
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    save_agent_step(
                        event_id=event_id,
                        agent_run=agent_run,
                        step=step,
                        action_type="tool_call",
                        tool_name=tc.get("name", ""),
                        tool_input=tc.get("args", {}),
                    )
            # Text reasoning
            elif hasattr(msg, "content") and msg.content:
                step += 1
                save_agent_step(
                    event_id=event_id,
                    agent_run=agent_run,
                    step=step,
                    action_type="reasoning",
                    reasoning=str(msg.content)[:2000],
                )
        except Exception as e:
            logger.debug("agent_step_persist_error", error=str(e))


def get_memory_context_for_chat(
    session_id: str,
    patient_id: Optional[str] = None,
    history_limit: int = 10,
) -> Dict:
    """
    Returns a dict with:
      - recent chat messages (for LLM context)
      - patient profile
      - recent events for this patient
    """
    chat_hist = get_chat_history(session_id, limit=history_limit,
                                  patient_id=patient_id)
    patient = get_patient(patient_id) if patient_id else None
    recent_events = []
    if patient_id:
        from backend.utils.memory_db import get_detection_events
        recent_events = get_detection_events(
            limit=10, patient_id=patient_id
        )

    return {
        "chat_history": chat_hist,
        "patient": patient,
        "recent_events": recent_events,
    }


def snapshot_system_state(video_processor=None, inference_engine=None,
                            ws_manager=None):
    """Periodically called to snapshot system health to DB."""
    import psutil
    cameras = {}
    if video_processor:
        cameras = video_processor.get_status()

    stats = get_event_stats(hours=1)

    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
    except Exception:
        cpu, mem = 0.0, 0.0

    save_system_snapshot(
        cameras=cameras,
        ws_clients=ws_manager.connection_count if ws_manager else 0,
        detections_hour=stats.get("total_events", 0),
        alerts_hour=stats.get("by_severity", {}).get("critical", 0),
        cpu_pct=cpu,
        mem_pct=mem,
    )


def _find_patient_by_camera(camera_id: str) -> Optional[str]:
    """Look up a patient associated with a camera ID."""
    try:
        for p in list_patients():
            if camera_id in p.get("camera_ids", []):
                return p["id"]
    except Exception:
        pass
    return None
