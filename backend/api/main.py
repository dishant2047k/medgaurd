"""
backend/api/main.py

FastAPI application entry point.
Starts video processing + inference engine as background tasks.
"""
from __future__ import annotations
import asyncio
import base64
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from backend.api.websocket_manager import manager
from backend.ml.pipeline.video_processor import VideoProcessor
from backend.ml.pipeline.inference_engine import InferenceEngine, MedicalEvent
from backend.agents.emergency_agent import trigger_emergency_response
from backend.rag.chat_assistant import MedicalChatAssistant
from backend.utils.config import get_settings
from backend.utils.database import init_db, get_db, DetectionEvent, Patient, ChatMessage
from backend.utils.logger import setup_logging, get_logger

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

# ── Global services ──────────────────────────────────────────

frame_queue = asyncio.Queue(maxsize=100)
video_processor: Optional[VideoProcessor] = None
inference_engine: Optional[InferenceEngine] = None
chat_assistant: Optional[MedicalChatAssistant] = None


# ── Event handlers ───────────────────────────────────────────

async def on_medical_event(event: MedicalEvent):
    """Callback fired when inference engine detects a medical event."""
    # 1. Push to WebSocket clients
    await manager.broadcast_all({
        "type": "medical_event",
        "event_type": event.event_type,
        "severity": event.severity,
        "confidence": round(event.confidence, 3),
        "camera_id": event.camera_id,
        "timestamp": event.timestamp,
        "snapshot": base64.b64encode(event.snapshot_bytes).decode()
                    if event.snapshot_bytes else None,
        "metadata": event.metadata,
    })

    # 2. Trigger emergency agent asynchronously (don't block)
    if event.severity in ("high", "critical"):
        asyncio.create_task(
            trigger_emergency_response(
                event_type=event.event_type,
                severity=event.severity,
                confidence=event.confidence,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                snapshot_bytes=event.snapshot_bytes,
                latitude=settings.default_latitude,
                longitude=settings.default_longitude,
            )
        )


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_processor, inference_engine, chat_assistant

    # DB
    await init_db()
    logger.info("database_initialised")

    # ML
    inference_engine = InferenceEngine(frame_queue)
    inference_engine.register_callback(on_medical_event)

    # Video
    video_processor = VideoProcessor(
        sources=settings.camera_source_list,
        frame_queue=frame_queue,
        target_fps=15,
        resolution=(640, 480),
    )
    loop = asyncio.get_running_loop()
    video_processor.start(loop)

    # Inference background task
    inference_task = asyncio.create_task(inference_engine.run())

    # RAG Chat
    chat_assistant = MedicalChatAssistant()

    logger.info("medguard_ai_started")
    yield

    # Cleanup
    video_processor.stop()
    inference_engine.stop()
    inference_task.cancel()
    logger.info("medguard_ai_stopped")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="MedGuard AI",
    description="Real-time Medical Emergency Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── WebSocket ────────────────────────────────────────────────

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await manager.connect(websocket, room="global")
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.websocket("/ws/camera/{camera_id}")
async def websocket_camera(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, room=camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


# ── REST: System ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cameras": video_processor.get_status() if video_processor else {},
        "ws_connections": manager.connection_count,
    }

@app.get("/api/camera/{camera_id}/stream")
async def camera_stream(camera_id: str):
    async def generate():
        import cv2
        while True:
            try:
                if video_processor:
                    frame = video_processor.get_latest_frame(camera_id)
                    if frame is not None:
                        _, buffer = cv2.imencode(
                            ".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
            except Exception:
                pass
            await asyncio.sleep(1 / 15)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
# ── REST: Patients ───────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    blood_group: Optional[str] = None
    conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    emergency_contacts: List[dict] = []
    camera_ids: List[str] = []


@app.post("/api/patients")
async def create_patient(data: PatientCreate, db=Depends(get_db)):
    patient = Patient(**data.model_dump())
    db.add(patient)
    await db.flush()
    return {"id": patient.id, "name": patient.name}


@app.get("/api/patients")
async def list_patients(db=Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Patient))
    patients = result.scalars().all()
    return [{"id": p.id, "name": p.name, "age": p.age, "conditions": p.conditions}
            for p in patients]


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, db=Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


# ── REST: Detections ─────────────────────────────────────────

@app.get("/api/detections")
async def list_detections(limit: int = 50, camera_id: Optional[str] = None,
                          db=Depends(get_db)):
    from sqlalchemy import select, desc
    q = select(DetectionEvent).order_by(desc(DetectionEvent.timestamp)).limit(limit)
    if camera_id:
        q = q.where(DetectionEvent.camera_id == camera_id)
    result = await db.execute(q)
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "severity": e.severity,
            "confidence": e.confidence,
            "camera_id": e.camera_id,
            "timestamp": e.timestamp.isoformat(),
            "resolved": e.resolved,
        }
        for e in events
    ]


@app.patch("/api/detections/{event_id}/resolve")
async def resolve_detection(event_id: str, db=Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(
        select(DetectionEvent).where(DetectionEvent.id == event_id)
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(404, "Event not found")
    event.resolved = True
    return {"resolved": True}


# ── REST: Chat ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str
    patient_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest, db=Depends(get_db)):
    patient_context = None
    if req.patient_id:
        from sqlalchemy import select
        result = await db.execute(
            select(Patient).where(Patient.id == req.patient_id)
        )
        p = result.scalar_one_or_none()
        if p:
            patient_context = {
                "name": p.name,
                "age": p.age,
                "conditions": p.conditions,
                "medications": p.medications,
                "allergies": p.allergies,
            }

    response = await chat_assistant.chat(
        message=req.message,
        session_id=req.session_id,
        patient_id=req.patient_id,
        patient_context=patient_context,
    )

    # Persist to DB
    db.add(ChatMessage(
        patient_id=req.patient_id,
        session_id=req.session_id,
        role="user",
        content=req.message,
    ))
    db.add(ChatMessage(
        patient_id=req.patient_id,
        session_id=req.session_id,
        role="assistant",
        content=response,
    ))

    return {"response": response, "session_id": req.session_id}


@app.post("/api/patients/{patient_id}/documents")
async def upload_patient_document(
    patient_id: str,
    file: UploadFile = File(...),
    db=Depends(get_db),
):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    await chat_assistant.ingest_patient_document(
        patient_id=patient_id,
        text=text,
        metadata={"filename": file.filename},
    )
    return {"status": "ingested", "filename": file.filename}


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
