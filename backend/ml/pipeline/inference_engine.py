"""
backend/ml/pipeline/inference_engine.py

Orchestrates all ML detectors into a single async pipeline.
Consumes FramePackets from the VideoProcessor queue and emits DetectionEvents.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Awaitable
import io

import cv2
import numpy as np

from backend.ml.models.pose_estimator import PoseEstimator
from backend.ml.models.fall_detector import FallDetector
from backend.ml.models.action_classifier import ActionClassifier
from backend.ml.models.facial_analyzer import FacialAnalyzer
from backend.ml.pipeline.video_processor import FramePacket
from backend.utils.config import get_settings
from backend.utils.logger import get_logger, DETECTION_COUNTER

logger = get_logger(__name__)
settings = get_settings()

# Minimum mean pixel brightness (0-255) to treat a frame as valid
_MIN_FRAME_BRIGHTNESS = 15
# Minimum fraction of non-zero pixels required (avoids mostly-black frames)
_MIN_NONZERO_RATIO = 0.05


@dataclass
class MedicalEvent:
    camera_id: str
    event_type: str        # fall | seizure | cardiac | facial_distress | unconscious | abnormal
    severity: str          # low | medium | high | critical
    confidence: float
    timestamp: float
    snapshot_bytes: Optional[bytes] = None
    metadata: dict = field(default_factory=dict)


EventCallback = Callable[[MedicalEvent], Awaitable[None]]


class _Cooldown:
    def __init__(self, seconds: int):
        self._last: Dict[str, float] = {}
        self._seconds = seconds

    def check(self, key: str) -> bool:
        """Returns True if enough time has passed since last trigger."""
        now = time.time()
        if now - self._last.get(key, 0) >= self._seconds:
            self._last[key] = now
            return True
        return False


class InferenceEngine:
    """
    Pulls FramePackets from queue, runs all detectors, and calls
    registered callbacks when medical events are detected.
    """

    def __init__(self, frame_queue: asyncio.Queue):
        self.frame_queue = frame_queue
        self._callbacks: List[EventCallback] = []
        self._cooldown = _Cooldown(settings.alert_cooldown_seconds)
        self._running = False

        self.pose_estimator = PoseEstimator(
            backend="yolo",
            model_path=settings.yolo_pose_model,
        )
        self.fall_detector = FallDetector()
        self.action_classifier = ActionClassifier()
        self.facial_analyzer = FacialAnalyzer()

        logger.info("inference_engine_initialised")

    def register_callback(self, cb: EventCallback):
        self._callbacks.append(cb)

    async def run(self):
        self._running = True
        logger.info("inference_engine_running")

        while self._running:
            try:
                packet: FramePacket = await asyncio.wait_for(
                    self.frame_queue.get(), timeout=1.0
                )
                await self._process(packet)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("inference_error", error=str(e), exc_info=True)

    def stop(self):
        self._running = False

    # ── Frame validation ─────────────────────────────────────

    @staticmethod
    def _is_valid_frame(frame: np.ndarray) -> bool:
        """
        Rejects None, empty, or blank/black frames.
        A valid frame must:
          - Not be None or empty
          - Have mean brightness above threshold (not a black frame)
          - Have enough non-zero pixels (not a mostly-black frame)
        """
        if frame is None or frame.size == 0:
            return False
        if np.mean(frame) < _MIN_FRAME_BRIGHTNESS:
            return False
        nonzero_ratio = np.count_nonzero(frame) / frame.size
        if nonzero_ratio < _MIN_NONZERO_RATIO:
            return False
        return True

    # ── Core processing ──────────────────────────────────────

    async def _process(self, packet: FramePacket):
        frame = packet.frame
        cam_id = packet.camera_id
        ts = packet.timestamp

        # ── FIX: Validate frame before running any detector ──────────────
        if not self._is_valid_frame(frame):
            logger.debug(
                "frame_skipped_invalid",
                camera_id=cam_id,
                frame_idx=packet.frame_idx,
            )
            return

        events: List[MedicalEvent] = []

        # 1. Pose estimation
        poses = self.pose_estimator.predict(frame)

        # 2. Fall / collapse detection
        if poses:
            fall_scores = self.fall_detector.update(poses, ts)
            for score in fall_scores:
                if score.is_fall and score.confidence > settings.detection_confidence:
                    events.append(MedicalEvent(
                        camera_id=cam_id,
                        event_type="fall",
                        severity=self._severity(score.confidence),
                        confidence=score.confidence,
                        timestamp=ts,
                        metadata={"reason": score.reason,
                                  "trunk_angle": score.trunk_angle},
                    ))

            # 3. Unconsciousness check (immobility after fall)
            for pose in poses:
                if self.fall_detector.check_immobility(pose.person_id):
                    events.append(MedicalEvent(
                        camera_id=cam_id,
                        event_type="unconscious",
                        severity="critical",
                        confidence=0.85,
                        timestamp=ts,
                        metadata={"person_id": pose.person_id},
                    ))

            # 4. Action classification (seizure / cardiac / tremor)
            if np.mean(frame) < 30:
                action_scores = []
            else:
                action_scores = self.action_classifier.update(poses, fps=15.0)
            for score in action_scores:
                if score.confidence > settings.detection_confidence:
                    events.append(MedicalEvent(
                        camera_id=cam_id,
                        event_type=score.event_type,
                        severity=self._severity(score.confidence),
                        confidence=score.confidence,
                        timestamp=ts,
                        metadata=score.details,
                    ))

        # 5. Facial distress analysis (every 5 frames for performance)
        if packet.frame_idx % 5 == 0 and np.mean(frame)>30:
            facial_results = self.facial_analyzer.analyse(frame)
            for res in facial_results:
                if res.distress_confidence > settings.detection_confidence:
                    events.append(MedicalEvent(
                        camera_id=cam_id,
                        event_type="facial_distress",
                        severity=self._severity(res.distress_confidence),
                        confidence=res.distress_confidence,
                        timestamp=ts,
                        metadata={
                            "emotion": res.dominant_emotion,
                            "eyes_open": res.eyes_open,
                            "asymmetry": res.asymmetry_score,
                        },
                    ))

        # Deduplicate, add snapshots, call callbacks
        for event in events:
            key = f"{cam_id}:{event.event_type}"
            if self._cooldown.check(key):
                event.snapshot_bytes = self._capture_snapshot(frame)
                DETECTION_COUNTER.labels(
                    event_type=event.event_type,
                    camera_id=cam_id,
                ).inc()
                logger.warning(
                    "medical_event_detected",
                    event_type=event.event_type,
                    confidence=round(event.confidence, 3),
                    camera_id=cam_id,
                )
                await self._dispatch(event)

    async def _dispatch(self, event: MedicalEvent):
        for cb in self._callbacks:
            try:
                await cb(event)
            except Exception as e:
                logger.error("callback_error", error=str(e))

    @staticmethod
    def _capture_snapshot(frame: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(buffer) if success else b""

    @staticmethod
    def _severity(confidence: float) -> str:
        if confidence >= 0.9:
            return "critical"
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.6:
            return "medium"
        return "low"