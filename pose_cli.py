"""Command-line entry point for Human Pose Estimation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.pose_estimator import PoseEstimator, process_image, process_video, process_webcam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect human body landmarks and visualize pose skeletons."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to an input image.")
    source.add_argument("--video", type=Path, help="Path to an input video.")
    source.add_argument("--webcam", action="store_true", help="Use the default webcam.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image/video path. Defaults to *_pose.jpg or *_pose.mp4."
    )
    parser.add_argument(
        "--json", dest="json_path", type=Path, default=None,
        help="Optional path for exporting detected landmark coordinates and joint angles."
    )
    parser.add_argument(
        "--model-complexity", type=int, choices=(0, 1, 2), default=1,
        help="MediaPipe model complexity: 0 (fast), 1 (balanced), 2 (accurate)."
    )
    parser.add_argument(
        "--detection-confidence", type=float, default=0.5,
        help="Minimum person detection confidence, from 0 to 1."
    )
    parser.add_argument(
        "--tracking-confidence", type=float, default=0.5,
        help="Minimum landmark tracking confidence, from 0 to 1."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name, value in (
        ("detection confidence", args.detection_confidence),
        ("tracking confidence", args.tracking_confidence),
    ):
        if not 0 <= value <= 1:
            print(f"Error: {name} must be between 0 and 1.", file=sys.stderr)
            return 2

    try:
        if args.image:
            if not args.image.is_file():
                raise FileNotFoundError(f"Image not found: {args.image}")
            output = args.output or args.image.with_name(f"{args.image.stem}_pose.jpg")
            result = process_image(
                args.image, output, args.json_path,
                args.model_complexity, args.detection_confidence,
                args.tracking_confidence,
            )
            print(f"Processed image saved to: {result}")
        elif args.video:
            if not args.video.is_file():
                raise FileNotFoundError(f"Video not found: {args.video}")
            output = args.output or args.video.with_name(f"{args.video.stem}_pose.mp4")
            result = process_video(
                args.video, output, args.json_path,
                args.model_complexity, args.detection_confidence,
                args.tracking_confidence,
            )
            print(f"Processed video saved to: {result}")
        else:
            result = process_webcam(
                args.output, args.json_path, args.model_complexity,
                args.detection_confidence, args.tracking_confidence,
            )
            if result:
                print(f"Webcam recording saved to: {result}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
