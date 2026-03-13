"""
scripts/test_camera.py

Test the video pipeline and pose detection on a camera source.
Usage:
  python scripts/test_camera.py --source 0           # webcam
  python scripts/test_camera.py --source video.mp4   # video file
  python scripts/test_camera.py --source rtsp://...  # CCTV stream
"""
import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from backend.ml.models.pose_estimator import PoseEstimator
from backend.ml.models.fall_detector import FallDetector
from backend.ml.models.action_classifier import ActionClassifier


def draw_poses(frame, poses, fall_scores, action_scores):
    """Draw skeleton, bboxes and labels on frame."""
    h, w = frame.shape[:2]

    for pose in poses:
        # Draw skeleton
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

        # Draw keypoints
        for kp in pose.keypoints:
            if kp.confidence > 0.3:
                pt = (int(kp.x * w), int(kp.y * h))
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)

        # Draw bbox
        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(frame,
                      (int(x1 * w), int(y1 * h)),
                      (int(x2 * w), int(y2 * h)),
                      (0, 100, 200), 1)

    # Draw detection labels
    y = 30
    for score in fall_scores:
        if score.confidence > 0.5:
            label = f"FALL: {score.confidence:.0%} [{score.reason}]"
            cv2.rectangle(frame, (5, y - 20), (len(label) * 9 + 10, y + 5), (255, 45, 85), -1)
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

    for score in action_scores:
        if score.confidence > 0.4:
            label = f"{score.event_type.upper()}: {score.confidence:.0%}"
            color = (255, 159, 10) if score.event_type == "seizure" else (255, 45, 85)
            cv2.rectangle(frame, (5, y - 20), (len(label) * 9 + 10, y + 5), color, -1)
            cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 30

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera source (0 = webcam)")
    parser.add_argument("--show", action="store_true", default=True, help="Display window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"🎥 Opening source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot open camera source")
        return

    pose_est = PoseEstimator()
    fall_det = FallDetector()
    action_cls = ActionClassifier()

    print("✅ Models loaded. Press 'q' to quit.")
    fps_counter = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        poses = pose_est.predict(frame)
        fall_scores = fall_det.update(poses, time.time())
        action_scores = action_cls.update(poses, fps=15.0)

        # Annotate
        frame = draw_poses(frame, poses, fall_scores, action_scores)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - t_start
        fps = fps_counter / elapsed
        cv2.putText(frame, f"FPS: {fps:.1f} | Persons: {len(poses)}",
                    (frame.shape[1] - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

        if args.show:
            cv2.imshow("MedGuard AI — Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera test complete")


if __name__ == "__main__":
    main()
