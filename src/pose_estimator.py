"""MediaPipe-based human pose detection, drawing, and joint-angle analysis."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark triplets used for the principal joint-angle measurements.
ANGLE_JOINTS = {
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
}


def calculate_angle(a: tuple[float, float], b: tuple[float, float],
                    c: tuple[float, float]) -> float:
    """Return the smaller angle ABC in degrees (0–180)."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def extract_pose_data(results: Any) -> dict[str, Any] | None:
    """Convert MediaPipe results into JSON-friendly landmarks and angles."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    points: dict[str, tuple[float, float]] = {}
    landmark_data: dict[str, dict[str, float]] = {}

    for landmark_id, landmark in enumerate(landmarks):
        name = mp_pose.PoseLandmark(landmark_id).name
        points[name] = (landmark.x, landmark.y)
        landmark_data[name] = {
            "x": round(float(landmark.x), 6),
            "y": round(float(landmark.y), 6),
            "z": round(float(landmark.z), 6),
            "visibility": round(float(landmark.visibility), 6),
        }

    angles: dict[str, float | None] = {}
    for joint, names in ANGLE_JOINTS.items():
        value = calculate_angle(*(points[name] for name in names))
        angles[joint] = round(value, 2) if math.isfinite(value) else None

    return {"landmarks": landmark_data, "joint_angles_degrees": angles}


def draw_pose(frame: np.ndarray, results: Any) -> np.ndarray:
    """Draw connected pose landmarks and readable joint-angle labels."""
    annotated = frame.copy()
    if not results.pose_landmarks:
        cv2.putText(
            annotated, "No person detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )
        return annotated

    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 220, 120), thickness=2, circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 180, 50), thickness=2, circle_radius=2
        ),
    )

    data = extract_pose_data(results)
    assert data is not None
    display = [
        ("L elbow", data["joint_angles_degrees"]["left_elbow"]),
        ("R elbow", data["joint_angles_degrees"]["right_elbow"]),
        ("L knee", data["joint_angles_degrees"]["left_knee"]),
        ("R knee", data["joint_angles_degrees"]["right_knee"]),
    ]
    y = 28
    for label, angle in display:
        value = f"{angle:.0f} deg" if angle is not None else "N/A"
        cv2.putText(
            annotated, f"{label}: {value}", (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 25
    return annotated


def _pose_config(model_complexity: int, detection_confidence: float,
                 tracking_confidence: float, static_image_mode: bool = False):
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=not static_image_mode,
        enable_segmentation=False,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


def _write_json(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def process_image(input_path: Path, output_path: Path, json_path: Path | None = None,
                  model_complexity: int = 1, detection_confidence: float = 0.5,
                  tracking_confidence: float = 0.5) -> Path:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise ValueError(f"Could not read image: {input_path}")
    with _pose_config(model_complexity, detection_confidence,
                      tracking_confidence, static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    annotated = draw_pose(frame, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write output image: {output_path}")
    _write_json(json_path, [extract_pose_data(results)] if results.pose_landmarks else [])
    return output_path


def process_video(input_path: Path, output_path: Path, json_path: Path | None = None,
                  model_complexity: int = 1, detection_confidence: float = 0.5,
                  tracking_confidence: float = 0.5) -> Path:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video has invalid dimensions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    records: list[dict[str, Any]] = []
    frame_number = 0
    try:
        with _pose_config(model_complexity, detection_confidence,
                          tracking_confidence) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                writer.write(draw_pose(frame, results))
                data = extract_pose_data(results)
                if data is not None:
                    records.append({"frame": frame_number, **data})
                frame_number += 1
    finally:
        capture.release()
        writer.release()

    if frame_number == 0:
        raise ValueError("No readable frames were found in the video.")
    _write_json(json_path, records)
    return output_path


def process_webcam(output_path: Path | None = None, json_path: Path | None = None,
                   model_complexity: int = 1, detection_confidence: float = 0.5,
                   tracking_confidence: float = 0.5) -> Path | None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not access the default webcam.")

    writer = None
    records: list[dict[str, Any]] = []
    frame_number = 0
    try:
        with _pose_config(model_complexity, detection_confidence,
                          tracking_confidence) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated = draw_pose(frame, results)
                if output_path is not None and writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create recording: {output_path}")
                if writer is not None:
                    writer.write(annotated)

                cv2.imshow("Human Pose Estimation | Press Q to quit", annotated)
                data = extract_pose_data(results)
                if data is not None:
                    records.append({"frame": frame_number, **data})
                frame_number += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    _write_json(json_path, records)
    return output_path if writer is not None else None
