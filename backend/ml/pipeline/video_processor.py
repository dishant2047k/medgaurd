"""
backend/ml/pipeline/video_processor.py

Real-time multi-camera video processing pipeline.
Manages camera capture, frame queuing, and inference dispatch.
Supports: webcam (index), RTSP (CCTV), file path (dashcam recordings).
"""
import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import cv2
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum mean pixel brightness to consider a frame worth queuing
_MIN_FRAME_BRIGHTNESS = 15
# How many consecutive blank frames before logging a warning
_BLANK_FRAME_WARN_THRESHOLD = 30


@dataclass
class FramePacket:
    """A single captured frame with metadata."""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    frame_idx: int


@dataclass
class CameraStream:
    """State for one camera feed."""
    camera_id: str
    source: Any                          # int or str (RTSP URL)
    cap: Optional[cv2.VideoCapture] = None
    running: bool = False
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    thread: Optional[threading.Thread] = None
    fps: float = 25.0
    reconnect_delay: float = 5.0
    error_count: int = 0
    blank_frame_count: int = 0           # consecutive blank frames


class VideoProcessor:
    """
    Manages N camera streams concurrently.
    Pushes FramePackets to an asyncio queue consumed by the inference engine.
    """

    def __init__(
        self,
        sources: List[Any],
        frame_queue: asyncio.Queue,
        target_fps: int = 15,
        resolution: tuple = (640, 480),
    ):
        self.sources = sources
        self.frame_queue = frame_queue
        self.target_fps = target_fps
        self.resolution = resolution
        self.streams: Dict[str, CameraStream] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._latest_frames: Dict[str, np.ndarray] = {}

    # ── Public API ──────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        for idx, src in enumerate(self.sources):
            cam_id = f"cam_{idx}" if not isinstance(src, str) or not src.startswith("rtsp") \
                     else f"cctv_{idx}"
            stream = CameraStream(camera_id=cam_id, source=src)
            self.streams[cam_id] = stream
            t = threading.Thread(
                target=self._capture_loop,
                args=(stream,),
                daemon=True,
                name=f"capture-{cam_id}",
            )
            stream.thread = t
            t.start()
            logger.info("camera_started", camera_id=cam_id, source=str(src))

    def stop(self):
        self._running = False
        for stream in self.streams.values():
            stream.running = False
            if stream.cap:
                stream.cap.release()
        logger.info("video_processor_stopped")

    def get_status(self) -> Dict:
        return {
            cam_id: {
                "running": s.running,
                "fps": s.fps,
                "buffer_size": len(s.frame_buffer),
                "errors": s.error_count,
                "blank_frames": s.blank_frame_count,
            }
            for cam_id, s in self.streams.items()
        }
    
    def get_latest_frame(self, camera_id: str):
        return self._latest_frames.get(camera_id)


    # ── Internal capture loop (runs in thread) ──────────────

    def _capture_loop(self, stream: CameraStream):
        frame_interval = 1.0 / self.target_fps
        frame_idx = 0

        while self._running:
            # Connect / reconnect
            if stream.cap is None or not stream.cap.isOpened():
                stream.cap = self._open_capture(stream.source)
                if stream.cap is None:
                    logger.warning("camera_connect_failed",
                                   camera_id=stream.camera_id,
                                   source=str(stream.source))
                    stream.error_count += 1
                    time.sleep(stream.reconnect_delay)
                    continue
                stream.running = True
                logger.info("camera_connected", camera_id=stream.camera_id)

            t0 = time.monotonic()
            ret, frame = stream.cap.read()

            if not ret:
                logger.warning("frame_read_failed", camera_id=stream.camera_id)
                stream.cap.release()
                stream.cap = None
                stream.error_count += 1
                continue

            # Resize to target resolution
            frame = cv2.resize(frame, self.resolution)

            # ── FIX: Skip blank / black frames — don't queue them ────────
            if np.mean(frame) < _MIN_FRAME_BRIGHTNESS:
                stream.blank_frame_count += 1
                if stream.blank_frame_count == _BLANK_FRAME_WARN_THRESHOLD:
                    logger.warning(
                        "camera_sending_blank_frames",
                        camera_id=stream.camera_id,
                        consecutive_blank=stream.blank_frame_count,
                    )
                # Still increment frame_idx so modulo-based sampling stays correct
                frame_idx += 1
                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue
            else:
                if stream.blank_frame_count > 0:
                    logger.info(
                        "camera_resumed_valid_frames",
                        camera_id=stream.camera_id,
                        blank_frames_skipped=stream.blank_frame_count,
                    )
                stream.blank_frame_count = 0

            packet = FramePacket(
                camera_id=stream.camera_id,
                frame=frame,
                timestamp=time.time(),
                frame_idx=frame_idx,
            )
            stream.frame_buffer.append(packet)
            self._latest_frames[stream.camera_id] = frame
            frame_idx += 1

            # Push to asyncio queue (thread-safe)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._enqueue(packet), self._loop
                )

            # Throttle to target FPS
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    async def _enqueue(self, packet: FramePacket):
        try:
            self.frame_queue.put_nowait(packet)

        except asyncio.QueueFull:
            pass  # Drop frame rather than block
    def _open_capture(self, source) -> Optional[cv2.VideoCapture]:
        """
        Try opening the camera with multiple backends.
        On Windows, built-in laptop cameras work best with CAP_MSMF (Media Foundation).
        Falls back to CAP_DSHOW (DirectShow) then the default backend.
        """
        # For integer sources (webcam index), try Windows-specific backends first
        backends = []
        if isinstance(source, int):
            backends = [
                (cv2.CAP_MSMF, "MSMF"),    # Media Foundation — best for built-in cameras
                (cv2.CAP_DSHOW, "DSHOW"),   # DirectShow — fallback
                (cv2.CAP_ANY, "ANY"),        # Let OpenCV decide
            ]
        else:
            backends = [(cv2.CAP_ANY, "ANY")]

        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(source, backend)
                if not cap.isOpened():
                    logger.debug("camera_backend_failed", backend=name, source=str(source))
                    cap.release()
                    continue
                
                # Warm up — read a test frame to confirm it's truly working
                ret, _ = cap.read()
                if not ret:
                    logger.debug("camera_backend_no_frame", backend=name, source=str(source))
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                logger.info("camera_opened", backend=name, source=str(source))
                return cap
            
            except Exception as e:
                logger.debug("open_capture_error", backend=name, error=str(e), source=str(source))

        logger.error("all_camera_backends_failed", source=str(source))
        return None