"""
scripts/test_camera.py

Enhanced test the video pipeline and pose detection on a camera source.

Fixes:
  1. Robust camera backend initialization (Windows MSMF → DSHOW → ANY)
  2. Proper frame validation (reject blank/black frames)
  3. Temporal validation (requires N consecutive frames for alerts)
  4. Model loading timeout and error handling
  5. GPU/CPU fallback for model loading
  6. Memory management (detection buffer reset mechanism)
  7. Error handling and camera reconnection
  8. Resource cleanup (proper release of camera)
  9. NON-BLOCKING INFERENCE - run on separate thread
  10. FRAME BUFFERING - display updates smoothly while inferring
  11. FASTER DISPLAY - skip inference frames for smoother playback

Usage:
  python scripts/test_camera.py --source 0                # webcam
  python scripts/test_camera.py --source 0 --no-models    # camera only (no ML)
  python scripts/test_camera.py --source 0 --debug        # verbose logging
  python scripts/test_camera.py --source 0 --cpu          # force CPU mode
  python scripts/test_camera.py --source 0 --skip-inference # display only
"""

import argparse
import sys
import time
import os
from collections import defaultdict, deque
from typing import Optional
import threading
import queue

import cv2
import numpy as np

sys.path.insert(0, ".")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Detection confidence thresholds
FALL_CONFIDENCE_THRESHOLD = 0.75
ACTION_CONFIDENCE_THRESHOLD = 0.70

# Temporal validation
MIN_CONSECUTIVE_FRAMES = 5
DETECTION_FRAME_TIMEOUT = 15

# Frame quality thresholds
MIN_FRAME_BRIGHTNESS = 15
MIN_NONZERO_RATIO = 0.05

# Camera configuration
DEFAULT_RESOLUTION = (640, 480)
TARGET_FPS = 15
FRAME_BUFFER_SIZE = 3  # Keep 3 frames in buffer

# Model loading timeout (30 seconds)
MODEL_LOAD_TIMEOUT = 30

# Inference configuration
INFERENCE_SKIP_FRAMES = 2  # Run inference every 2 frames (faster display)
INFERENCE_QUEUE_SIZE = 2  # Keep 2 frames in inference queue


# ═══════════════════════════════════════════════════════════════
# DETECTION BUFFER CLASS
# ═══════════════════════════════════════════════════════════════

class TemporalDetectionBuffer:
    """
    Temporal validation buffer for detections.
    Ensures events are detected for N consecutive frames before alerting.
    """
    
    def __init__(self, min_frames: int = 5, timeout_frames: int = 15):
        self.min_frames = min_frames
        self.timeout_frames = timeout_frames
        self.buffers = defaultdict(list)
        self.last_frame_idx = defaultdict(int)
        self.lock = threading.Lock()
    
    def add_detection(self, key: str, confidence: float, frame_idx: int) -> bool:
        """Add detection to buffer. Returns True if confirmed."""
        with self.lock:
            if (frame_idx - self.last_frame_idx[key]) > self.timeout_frames:
                self.buffers[key] = []
            
            self.buffers[key].append(confidence)
            self.last_frame_idx[key] = frame_idx
            
            if len(self.buffers[key]) > self.min_frames * 2:
                self.buffers[key] = self.buffers[key][-self.min_frames * 2:]
            
            return len(self.buffers[key]) >= self.min_frames
    
    def get_count(self, key: str) -> int:
        """Get current detection count in buffer."""
        with self.lock:
            return len(self.buffers[key])
    
    def reset_all(self):
        """Clear all buffers."""
        with self.lock:
            self.buffers.clear()
            self.last_frame_idx.clear()


# ═══════════════════════════════════════════════════════════════
# CAMERA INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def open_camera_with_fallback(
    source,
    resolution: tuple = DEFAULT_RESOLUTION,
    debug: bool = False
) -> Optional[cv2.VideoCapture]:
    """
    Try multiple backends to open camera.
    Windows: MSMF → DSHOW → ANY
    Linux: V4L2 → ANY
    """
    backends = []
    
    if isinstance(source, int):
        if sys.platform == "win32":
            backends = [
                (cv2.CAP_MSMF, "MSMF (Media Foundation)"),
                (cv2.CAP_DSHOW, "DSHOW (DirectShow)"),
                (cv2.CAP_ANY, "ANY (OpenCV default)"),
            ]
        elif sys.platform == "linux":
            backends = [
                (cv2.CAP_V4L2, "V4L2 (Video4Linux)"),
                (cv2.CAP_ANY, "ANY (OpenCV default)"),
            ]
        else:
            backends = [(cv2.CAP_ANY, "ANY (OpenCV default)")]
    else:
        backends = [(cv2.CAP_ANY, "ANY (OpenCV default)")]
    
    for backend_id, backend_name in backends:
        try:
            if debug:
                print(f"  ⏳ Trying {backend_name}...")
            
            cap = cv2.VideoCapture(source, backend_id)
            
            if not cap.isOpened():
                cap.release()
                if debug:
                    print(f"    ❌ Failed to open with {backend_name}")
                continue
            
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                if debug:
                    print(f"    ❌ Cannot read frames with {backend_name}")
                continue
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            
            if debug:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"    ✅ Opened with {backend_name}")
                print(f"       Resolution: {actual_w}x{actual_h}")
                print(f"       FPS: {actual_fps}")
            
            return cap
        
        except Exception as e:
            if debug:
                print(f"    ❌ {backend_name} error: {str(e)}")
            continue
    
    return None


# ═══════════════════════════════════════════════════════════════
# FRAME VALIDATION
# ═══════════════════════════════════════════════════════════════

def is_valid_frame(frame: np.ndarray) -> bool:
    """Validate frame quality before inference."""
    if frame is None or frame.size == 0:
        return False
    
    if np.mean(frame) < MIN_FRAME_BRIGHTNESS:
        return False
    
    nonzero_ratio = np.count_nonzero(frame) / frame.size
    if nonzero_ratio < MIN_NONZERO_RATIO:
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING WITH TIMEOUT
# ═══════════════════════════════════════════════════════════════

class ModelLoader:
    """Load ML models with timeout and error handling."""
    
    def __init__(self, debug: bool = False, use_cpu: bool = False):
        self.debug = debug
        self.use_cpu = use_cpu
        self.pose_est = None
        self.fall_det = None
        self.action_cls = None
        self.load_errors = []
    
    def load_models(self) -> bool:
        """Load all models with timeout. Returns True if successful."""
        
        if self.use_cpu:
            print("💾 Forcing CPU mode...")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Load Pose Estimator
        print("1️⃣  Loading Pose Estimator...")
        try:
            self.pose_est = self._load_with_timeout(
                self._load_pose_estimator, 
                "PoseEstimator",
                MODEL_LOAD_TIMEOUT
            )
            if self.pose_est is None:
                return False
            print("   ✅ Pose Estimator loaded\n")
        except Exception as e:
            self.load_errors.append(f"PoseEstimator: {str(e)}")
            print(f"   ❌ PoseEstimator failed: {str(e)}\n")
            return False
        
        # Load Fall Detector
        print("2️⃣  Loading Fall Detector...")
        try:
            self.fall_det = self._load_with_timeout(
                self._load_fall_detector,
                "FallDetector",
                MODEL_LOAD_TIMEOUT
            )
            if self.fall_det is None:
                return False
            print("   ✅ Fall Detector loaded\n")
        except Exception as e:
            self.load_errors.append(f"FallDetector: {str(e)}")
            print(f"   ❌ FallDetector failed: {str(e)}\n")
            return False
        
        # Load Action Classifier
        print("3️⃣  Loading Action Classifier...")
        try:
            self.action_cls = self._load_with_timeout(
                self._load_action_classifier,
                "ActionClassifier",
                MODEL_LOAD_TIMEOUT
            )
            if self.action_cls is None:
                return False
            print("   ✅ Action Classifier loaded\n")
        except Exception as e:
            self.load_errors.append(f"ActionClassifier: {str(e)}")
            print(f"   ❌ ActionClassifier failed: {str(e)}\n")
            return False
        
        return True
    
    def _load_with_timeout(self, load_func, name: str, timeout: int):
        """Load model with timeout."""
        result = [None]
        error = [None]
        
        def load_thread():
            try:
                if self.debug:
                    print(f"   ⏳ Loading {name} (this may take 15-30 seconds)...")
                result[0] = load_func()
                if self.debug:
                    print(f"   ✓ {name} loaded")
            except Exception as e:
                error[0] = e
                if self.debug:
                    print(f"   ✗ {name} error: {str(e)}")
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            print(f"   ⏱️  {name} loading timeout ({timeout}s) - skipping")
            return None
        
        if error[0]:
            raise error[0]
        
        return result[0]
    
    @staticmethod
    def _load_pose_estimator():
        """Load PoseEstimator model."""
        from backend.ml.models.pose_estimator import PoseEstimator
        return PoseEstimator()
    
    @staticmethod
    def _load_fall_detector():
        """Load FallDetector model."""
        from backend.ml.models.fall_detector import FallDetector
        return FallDetector()
    
    @staticmethod
    def _load_action_classifier():
        """Load ActionClassifier model."""
        from backend.ml.models.action_classifier import ActionClassifier
        return ActionClassifier()


# ═══════════════════════════════════════════════════════════════
# ASYNC INFERENCE WORKER (FIX: Non-blocking inference)
# ═══════════════════════════════════════════════════════════════

class InferenceWorker:
    """
    Run inference on separate thread to avoid blocking display.
    FIX: Display updates while inference runs in background.
    """
    
    def __init__(self, pose_est, fall_det, action_cls, debug=False):
        self.pose_est = pose_est
        self.fall_det = fall_det
        self.action_cls = action_cls
        self.debug = debug
        
        self.frame_queue = queue.Queue(maxsize=INFERENCE_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=INFERENCE_QUEUE_SIZE)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start inference worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop inference worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def submit_frame(self, frame, frame_idx):
        """Submit frame for inference (non-blocking)."""
        try:
            self.frame_queue.put_nowait((frame, frame_idx))
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def get_results(self):
        """Get latest inference results (non-blocking)."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _run(self):
        """Inference loop (runs on separate thread)."""
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame, frame_idx = self.frame_queue.get(timeout=0.5)
                
                # Run inference
                poses = self.pose_est.predict(frame)
                fall_scores = self.fall_det.update(poses, time.time())
                action_scores = self.action_cls.update(poses, fps=TARGET_FPS)
                
                # Put results in queue
                try:
                    self.result_queue.put_nowait({
                        'frame_idx': frame_idx,
                        'poses': poses,
                        'fall_scores': fall_scores,
                        'action_scores': action_scores,
                    })
                except queue.Full:
                    pass  # Drop result if queue is full
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug:
                    print(f"❌ Inference error: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def draw_poses(frame, poses, fall_scores, action_scores, detection_buffer_obj=None):
    """Draw skeleton, bboxes and labels on frame."""
    h, w = frame.shape[:2]

    # Draw skeletons
    for pose in poses:
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        kp_dict = {kp.name: kp for kp in pose.keypoints if kp.confidence > 0.3}

        for a, b in connections:
            if a in kp_dict and b in kp_dict:
                pt1 = (int(kp_dict[a].x * w), int(kp_dict[a].y * h))
                pt2 = (int(kp_dict[b].x * w), int(kp_dict[b].y * h))
                cv2.line(frame, pt1, pt2, (0, 229, 255), 2)

        for kp in pose.keypoints:
            if kp.confidence > 0.3:
                pt = (int(kp.x * w), int(kp.y * h))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)

        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(frame,
                      (int(x1 * w), int(y1 * h)),
                      (int(x2 * w), int(y2 * h)),
                      (0, 100, 200), 2)

    # Draw detection labels
    y = 30

    for score in fall_scores:
        if score.confidence > FALL_CONFIDENCE_THRESHOLD:
            label = f"🚨 FALL: {score.confidence:.0%}"
            cv2.rectangle(frame, (5, y - 20), (len(label) * 9 + 10, y + 5), (0, 0, 255), -1)
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

    for score in action_scores:
        if score.confidence > ACTION_CONFIDENCE_THRESHOLD:
            icon = "⚡" if score.event_type == "seizure" else "❤️"
            label = f"{icon} {score.event_type.upper()}: {score.confidence:.0%}"
            color = (255, 159, 10) if score.event_type == "seizure" else (0, 165, 255)
            cv2.rectangle(frame, (5, y - 20), (len(label) * 9 + 10, y + 5), color, -1)
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

    # Draw detection buffer status
    if detection_buffer_obj:
        y += 10
        cv2.putText(frame, "Detection Buffer:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        for det_key in list(detection_buffer_obj.buffers.keys())[:3]:  # Show top 3
            count = detection_buffer_obj.get_count(det_key)
            status = "✅" if count >= MIN_CONSECUTIVE_FRAMES else "⏳"
            label = f"{status} {det_key}: {count}/{MIN_CONSECUTIVE_FRAMES}"
            color = (0, 255, 0) if count >= MIN_CONSECUTIVE_FRAMES else (200, 200, 200)
            cv2.putText(frame, label, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 18

    return frame


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MedGuard AI — Camera Test & Detection"
    )
    parser.add_argument("--source", default="0", help="Camera source (0 = webcam)")
    parser.add_argument("--show", action="store_true", default=True, help="Display window")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--no-models", action="store_true", help="Skip model loading (camera only)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (no GPU)")
    parser.add_argument("--skip-inference", action="store_true", help="Display frames only, no inference")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print("\n" + "=" * 70)
    print("🏥 MedGuard AI — Camera Test & Detection")
    print("=" * 70)
    print(f"📹 Source: {source}")
    print(f"🔧 Debug: {'ON' if args.debug else 'OFF'}")
    print(f"🚫 Skip Models: {'YES' if args.no_models else 'NO'}")
    print(f"⏩ Skip Inference: {'YES' if args.skip_inference else 'NO'}")
    print(f"💾 CPU Only: {'YES' if args.cpu else 'NO'}")
    print("⌨️  Press Q to quit\n")

    # Initialize Camera
    print("📷 Initializing camera...")
    cap = open_camera_with_fallback(source, resolution=DEFAULT_RESOLUTION, debug=args.debug)

    if cap is None:
        print("❌ Failed to open camera")
        return 1

    print("✅ Camera opened successfully\n")

    # Initialize Models
    pose_est = None
    fall_det = None
    action_cls = None
    inference_worker = None

    if not args.no_models and not args.skip_inference:
        print("🧠 Loading ML models...\n")
        loader = ModelLoader(debug=args.debug, use_cpu=args.cpu)
        
        if loader.load_models():
            pose_est = loader.pose_est
            fall_det = loader.fall_det
            action_cls = loader.action_cls
            print("✅ All models loaded successfully\n")
            
            # Start inference worker (non-blocking)
            print("⚙️  Starting inference worker thread...\n")
            inference_worker = InferenceWorker(pose_est, fall_det, action_cls, debug=args.debug)
            inference_worker.start()
        else:
            print("\n⚠️  Model loading failed. Running in camera-only mode.")
            print("   Run with --cpu flag if GPU is causing issues.\n")
    else:
        if args.skip_inference:
            print("⏭️  Skipping inference (display-only mode)\n")
        else:
            print("⏭️  Skipping model loading (camera-only mode)\n")

    # Initialize Detection Buffer
    detection_buffer = TemporalDetectionBuffer(
        min_frames=MIN_CONSECUTIVE_FRAMES,
        timeout_frames=DETECTION_FRAME_TIMEOUT
    )

    # Main Loop
    print("▶️  Starting inference (Press Q to quit)...\n")

    fps_counter = 0
    t_start = time.time()
    frame_idx = 0
    error_count = 0
    max_errors = 10
    skip_counter = 0
    
    # Latest inference results (to display while new inference runs)
    latest_poses = []
    latest_fall_scores = []
    latest_action_scores = []

    try:
        while True:
            try:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    error_count += 1
                    if args.debug:
                        print(f"⚠️  Frame read failed (attempt {error_count}/{max_errors})")
                    
                    if error_count >= max_errors:
                        print("❌ Too many read errors. Attempting to reconnect...")
                        cap.release()
                        cap = open_camera_with_fallback(source, debug=args.debug)
                        if cap is None:
                            print("❌ Failed to reconnect camera")
                            break
                        error_count = 0
                        detection_buffer.reset_all()
                    continue
                
                error_count = 0
                
                # Validate Frame
                if not is_valid_frame(frame):
                    if args.debug:
                        print(f"⏭️  Skipping invalid frame {frame_idx}")
                    frame_idx += 1
                    continue

                # ── NON-BLOCKING INFERENCE ──────────────────────────────
                # Submit frame for inference (doesn't block)
                if inference_worker and skip_counter % INFERENCE_SKIP_FRAMES == 0:
                    inference_worker.submit_frame(frame.copy(), frame_idx)
                
                # Get latest inference results (if available)
                results = inference_worker.get_results() if inference_worker else None
                if results:
                    latest_poses = results['poses']
                    latest_fall_scores = results['fall_scores']
                    latest_action_scores = results['action_scores']
                    
                    # Temporal Validation (update detection buffer)
                    for score in latest_fall_scores:
                        if score.confidence > FALL_CONFIDENCE_THRESHOLD:
                            if detection_buffer.add_detection("fall", score.confidence, frame_idx):
                                print(f"✅ CONFIRMED FALL: {score.confidence:.0%}")
                    
                    for score in latest_action_scores:
                        if score.confidence > ACTION_CONFIDENCE_THRESHOLD:
                            det_key = f"action_{score.event_type}"
                            if detection_buffer.add_detection(det_key, score.confidence, frame_idx):
                                print(f"✅ CONFIRMED {score.event_type.upper()}: {score.confidence:.0%}")

                # ── ANNOTATE & DISPLAY (always updates, no blocking) ──
                if inference_worker:
                    frame = draw_poses(frame, latest_poses, latest_fall_scores, latest_action_scores, detection_buffer)

                # FPS counter
                fps_counter += 1
                elapsed = time.time() - t_start
                fps = fps_counter / elapsed if elapsed > 0 else 0
                status = "🧠 INFERENCE" if inference_worker else "📷 CAMERA ONLY"
                cv2.putText(
                    frame,
                    f"{status} | FPS: {fps:.1f} | Persons: {len(latest_poses)} | Frame: {frame_idx}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 229, 255),
                    2
                )

                # Display
                if args.show:
                    cv2.imshow("MedGuard AI — Press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_idx += 1
                skip_counter += 1

            except KeyboardInterrupt:
                print("\n⏹️  Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                error_count += 1
                continue

    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("🛑 Shutting down...")
        
        if inference_worker:
            inference_worker.stop()
        
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - t_start
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        print(f"\n📈 Statistics:")
        print(f"   Total frames: {frame_idx}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("\n👋 Camera test complete")
        print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())