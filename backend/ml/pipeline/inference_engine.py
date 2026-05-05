"""
backend/ml/pipeline/inference_engine.py

Orchestrates all ML detectors into a single async pipeline.
Consumes FramePackets from the VideoProcessor queue and emits DetectionEvents.

FIXES:
  1. Increased detection thresholds (0.85+ for fall, 0.80+ for actions)
  2. Temporal validation - requires multiple consecutive frames
  3. Frame brightness check - skips dark/black frames
  4. Cooldown mechanism - prevents alert spam
  5. Better severity mapping
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set
from collections import defaultdict

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

# ═══════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS (FIXED: Higher values = fewer false positives)
# ═══════════════════════════════════════════════════════════════

FALL_CONFIDENCE_THRESHOLD = 0.85  # ✅ INCREASED (was ~0.5)
ACTION_CONFIDENCE_THRESHOLD = 0.80  # ✅ INCREASED (was ~0.4)
FACIAL_CONFIDENCE_THRESHOLD = 0.75  # ✅ INCREASED (was ~0.5)
SEIZURE_CONFIDENCE_THRESHOLD = 0.85
CARDIAC_CONFIDENCE_THRESHOLD = 0.92
TREMOR_CONFIDENCE_THRESHOLD = 0.80
UNCONSCIOUS_CONFIDENCE_THRESHOLD = 0.90

# Temporal validation
MIN_CONSECUTIVE_FRAMES = 5
DETECTION_TIMEOUT_FRAMES = 30


@dataclass
class MedicalEvent:
    camera_id: str
    event_type: str        # fall | seizure | cardiac | tremor | facial_distress | unconscious | abnormal
    severity: str          # low | medium | high | critical
    confidence: float
    timestamp: float
    snapshot_bytes: Optional[bytes] = None
    metadata: dict = field(default_factory=dict)


EventCallback = Callable[[MedicalEvent], Awaitable[None]]


class TemporalDetectionBuffer:
    """
    Temporal validation buffer for detections.
    Ensures events are detected for N consecutive frames before alerting.
    Prevents single-frame false positives.
    """
    
    def __init__(self, min_frames: int = 5, timeout_frames: int = 30):
        self.min_frames = min_frames
        self.timeout_frames = timeout_frames
        self.buffers = defaultdict(list)
        self.last_frame_idx = defaultdict(int)
        self.confirmed: Set[str] = set()
    
    def add_detection(self, key: str, confidence: float, frame_idx: int) -> bool:
        """Add detection to buffer. Returns True if confirmed."""
        # Reset if timeout exceeded
        if frame_idx - self.last_frame_idx[key] > self.timeout_frames:
            self.reset(key)

        self.buffers[key].append(confidence)
        self.last_frame_idx[key] = frame_idx

        # Keep only recent detections (memory management)
        if len(self.buffers[key]) > self.min_frames * 2:
            self.buffers[key] = self.buffers[key][-self.min_frames * 2:]

        # Only fire once per continuous detection episode.
        if key in self.confirmed:
            return False

        # Return True only if enough consecutive frames AND high average confidence
        if len(self.buffers[key]) >= self.min_frames:
            recent = self.buffers[key][-self.min_frames:]
            avg_confidence = np.mean(recent)
            if avg_confidence > 0.75:
                self.confirmed.add(key)
                return True

        return False
    
    def get_count(self, key: str) -> int:
        """Get current detection count in buffer."""
        return len(self.buffers[key])
    
    def reset(self, key: str):
        """Clear buffer for a detection type."""
        self.buffers.pop(key, None)
        self.last_frame_idx.pop(key, None)
        self.confirmed.discard(key)

    def prune(self, frame_idx: int):
        """Drop stale detections so they can be re-armed after inactivity."""
        expired = [
            key for key, last_seen in self.last_frame_idx.items()
            if frame_idx - last_seen > self.timeout_frames
        ]
        for key in expired:
            self.reset(key)


class _Cooldown:
    """Prevents alert spam by enforcing minimum time between alerts."""
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
    
    Features:
      - Multi-detector pipeline (pose, fall, action, facial)
      - Temporal validation (requires N consecutive frames)
      - Frame quality validation (brightness, non-zero pixels)
      - Cooldown mechanism (prevents alert spam)
      - High detection thresholds (reduces false positives)
    """

    def __init__(self, frame_queue: asyncio.Queue):
        self.frame_queue = frame_queue
        self._callbacks: List[EventCallback] = []
        self._cooldown = _Cooldown(settings.alert_cooldown_seconds)
        self._running = False
        
        # Temporal validation buffer
        self.detection_buffer = TemporalDetectionBuffer(
            min_frames=MIN_CONSECUTIVE_FRAMES,
            timeout_frames=DETECTION_TIMEOUT_FRAMES
        )

        # Load all models
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
        
        brightness = np.mean(frame)
        if brightness < _MIN_FRAME_BRIGHTNESS:
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
        frame_idx = packet.frame_idx
        self.detection_buffer.prune(frame_idx)

        # ── FIX: Validate frame before running any detector ──────────────
        if not self._is_valid_frame(frame):
            logger.debug(
                "frame_skipped_invalid",
                camera_id=cam_id,
                frame_idx=frame_idx,
            )
            return

        events: List[MedicalEvent] = []
        frame_brightness = np.mean(frame)

        try:
            # 1. Pose estimation
            poses = self.pose_estimator.predict(frame)
            for pose in poses:
                pose.person_id = self._namespace_person_id(cam_id, pose.person_id)

            if not poses:
                return

            # 2. Fall / collapse detection
            fall_scores = self.fall_detector.update(poses, ts)
            for score in fall_scores:
                # FIX: Use higher threshold for fall detection
                if score.is_fall and score.confidence > FALL_CONFIDENCE_THRESHOLD:
                    det_key = f"fall_{cam_id}_{score.person_id}"
                    # Temporal validation: require N consecutive frames
                    if self.detection_buffer.add_detection(det_key, score.confidence, frame_idx):
                        events.append(MedicalEvent(
                            camera_id=cam_id,
                            event_type="fall",
                            severity=self._severity(score.confidence),
                            confidence=score.confidence,
                            timestamp=ts,
                            metadata={
                                "person_id": score.person_id,
                                "reason": score.reason,
                                "trunk_angle": score.trunk_angle,
                                "consecutive_frames": self.detection_buffer.get_count(det_key)
                            },
                        ))
                        logger.warning(
                            "medical_event_detected",
                            event_type="fall",
                            confidence=round(score.confidence, 3),
                            camera_id=cam_id,
                        )

            # 3. Unconsciousness check (immobility after fall)
            # FIX: Only alert if high confidence
            for pose in poses:
                if self.fall_detector.check_immobility(pose.person_id):
                    det_key = f"unconscious_{cam_id}_{pose.person_id}"
                    # Temporal validation
                    if self.detection_buffer.add_detection(det_key, 0.90, frame_idx):
                        events.append(MedicalEvent(
                            camera_id=cam_id,
                            event_type="unconscious",
                            severity="critical",
                            confidence=0.90,
                            timestamp=ts,
                            metadata={
                                "person_id": pose.person_id,
                                "consecutive_frames": self.detection_buffer.get_count(det_key),
                            },
                        ))
                        logger.warning(
                            "medical_event_detected",
                            event_type="unconscious",
                            confidence=0.90,
                            camera_id=cam_id,
                        )

            # 4. Action classification (seizure / cardiac / tremor)
            # FIX: Skip if frame is too dark (likely to be false positives)
            if frame_brightness > 30:
                action_scores = self.action_classifier.update(poses, fps=15.0)
                for score in action_scores:
                    # FIX: Use higher threshold for action classification
                    threshold = self._get_threshold_for_action(score.event_type)
                    if score.confidence > threshold:
                        det_key = f"action_{cam_id}_{score.person_id}_{score.event_type}"
                        # Temporal validation
                        if self.detection_buffer.add_detection(det_key, score.confidence, frame_idx):
                            events.append(MedicalEvent(
                                camera_id=cam_id,
                                event_type=score.event_type,
                                severity=self._severity(score.confidence),
                                confidence=score.confidence,
                                timestamp=ts,
                                metadata={
                                    "person_id": score.person_id,
                                    **score.details,
                                    "consecutive_frames": self.detection_buffer.get_count(det_key)
                                },
                            ))
                            logger.warning(
                                "medical_event_detected",
                                event_type=score.event_type,
                                confidence=round(score.confidence, 3),
                                camera_id=cam_id,
                            )

            # 5. Facial distress analysis (every 5 frames for performance)
            # FIX: Only run if frame is bright enough
            if frame_idx % 5 == 0 and frame_brightness > 30:
                facial_results = self.facial_analyzer.analyse(frame)
                for res in facial_results:
                    # FIX: Use higher threshold for facial analysis
                    if res.distress_confidence > FACIAL_CONFIDENCE_THRESHOLD:
                        det_key = f"facial_distress_{cam_id}_{res.person_id}"
                        # Temporal validation
                        if self.detection_buffer.add_detection(det_key, res.distress_confidence, frame_idx):
                            events.append(MedicalEvent(
                                camera_id=cam_id,
                                event_type="facial_distress",
                                severity=self._severity(res.distress_confidence),
                                confidence=res.distress_confidence,
                                timestamp=ts,
                                metadata={
                                    "person_id": res.person_id,
                                    "emotion": res.dominant_emotion,
                                    "eyes_open": res.eyes_open,
                                    "asymmetry": res.asymmetry_score,
                                    "consecutive_frames": self.detection_buffer.get_count(det_key),
                                },
                            ))
                            logger.warning(
                                "medical_event_detected",
                                event_type="facial_distress",
                                confidence=round(res.distress_confidence, 3),
                                camera_id=cam_id,
                            )

        except Exception as e:
            logger.error("detection_pipeline_error", error=str(e), exc_info=True)

        # Deduplicate, add snapshots, call callbacks
        for event in events:
            key = f"{event.camera_id}:{event.event_type}"
            if self._cooldown.check(key):
                event.snapshot_bytes = self._capture_snapshot(frame)
                DETECTION_COUNTER.labels(
                    event_type=event.event_type,
                    camera_id=event.camera_id,
                ).inc()
                await self._dispatch(event)

    async def _dispatch(self, event: MedicalEvent):
        """Call all registered callbacks for this event."""
        for cb in self._callbacks:
            try:
                await cb(event)
            except Exception as e:
                logger.error("callback_error", error=str(e))

    @staticmethod
    def _capture_snapshot(frame: np.ndarray) -> bytes:
        """Capture frame snapshot as JPEG bytes."""
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(buffer) if success else b""

    @staticmethod
    def _get_threshold_for_action(event_type: str) -> float:
        """Get specific threshold for each action type."""
        thresholds = {
            "seizure": SEIZURE_CONFIDENCE_THRESHOLD,
            "cardiac": CARDIAC_CONFIDENCE_THRESHOLD,
            "tremor": TREMOR_CONFIDENCE_THRESHOLD,
        }
        return thresholds.get(event_type, ACTION_CONFIDENCE_THRESHOLD)

    @staticmethod
    def _severity(confidence: float) -> str:
        """Map confidence to severity level."""
        if confidence >= 0.95:
            return "critical"
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.70:
            return "medium"
        return "low"

    @staticmethod
    def _namespace_person_id(camera_id: str, person_id: int) -> int:
        """
        Scope transient detector person ids to a camera so per-person buffers
        do not bleed across feeds that all start numbering at zero.
        """
        camera_slot = sum((idx + 1) * ord(ch) for idx, ch in enumerate(camera_id))
        return camera_slot * 1000 + person_id
