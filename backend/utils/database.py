"""
backend/utils/database.py
Async SQLAlchemy setup with all ORM models.
"""
from datetime import datetime
from typing import AsyncGenerator
import uuid
import os

from sqlalchemy import (
    String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Enum as SAEnum
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum

from backend.utils.config import get_settings

settings = get_settings()

_db_url = settings.database_url
if _db_url.startswith("sqlite") and ":memory:" not in _db_url:
    sqlite_path = _db_url.split("///", 1)[-1]
    sqlite_dir = os.path.dirname(os.path.abspath(sqlite_path))
    if sqlite_dir:
        os.makedirs(sqlite_dir, exist_ok=True)

engine = create_async_engine(
    _db_url,
    echo=settings.app_env == "development",
    connect_args={"check_same_thread": False} if "sqlite" in _db_url else {},
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class Base(DeclarativeBase):
    pass


class EventTypeEnum(str, enum.Enum):
    FALL = "fall"
    SEIZURE = "seizure"
    CARDIAC = "cardiac"
    UNCONSCIOUS = "unconscious"
    FACIAL_DISTRESS = "facial_distress"
    ABNORMAL_MOVEMENT = "abnormal_movement"
    UNKNOWN = "unknown"


class SeverityEnum(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(200))
    age: Mapped[int] = mapped_column()
    gender: Mapped[str] = mapped_column(String(20), nullable=True)
    blood_group: Mapped[str] = mapped_column(String(10), nullable=True)
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    medications: Mapped[list] = mapped_column(JSON, default=list)
    allergies: Mapped[list] = mapped_column(JSON, default=list)
    emergency_contacts: Mapped[list] = mapped_column(JSON, default=list)
    camera_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                   onupdate=datetime.utcnow)

    detections: Mapped[list["DetectionEvent"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=True)
    camera_id: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[str] = mapped_column(SAEnum(EventTypeEnum))
    severity: Mapped[str] = mapped_column(SAEnum(SeverityEnum))
    confidence: Mapped[float] = mapped_column(Float)
    event_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    snapshot_path: Mapped[str] = mapped_column(String(500), nullable=True)
    video_clip_path: Mapped[str] = mapped_column(String(500), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    alert_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped["Patient"] = relationship(back_populates="detections")
    alerts: Mapped[list["AlertLog"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(ForeignKey("detection_events.id"))
    channel: Mapped[str] = mapped_column(String(50))
    recipient: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(50))
    response: Mapped[str] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    event: Mapped["DetectionEvent"] = relationship(back_populates="alerts")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
