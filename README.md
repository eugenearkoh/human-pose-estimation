# Human Pose Estimation

A Python computer-vision project that detects human body landmarks from images, video files, or a live webcam feed using **MediaPipe Pose** and **OpenCV**. It draws a connected human skeleton, tracks landmarks across video frames, calculates selected joint angles, and can export landmark coordinates and angle measurements to JSON.

## Features

- Detects 33 MediaPipe body landmarks.
- Highlights shoulders, elbows, wrists, hips, knees, and other pose landmarks.
- Draws the connected human skeleton over the original image or video.
- Calculates left/right shoulder, elbow, hip, and knee angles.
- Processes still images, recorded videos, and a live webcam.
- Exports normalized landmark coordinates (`x`, `y`, `z`, visibility) and joint angles to JSON.
- Configurable detection confidence, tracking confidence, and model complexity.

## Requirements

- Python 3.10 or 3.11 (recommended)
- A webcam for live-camera mode
- Windows, macOS, or Linux

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/human-pose-estimation.git
cd human-pose-estimation
```

Create and activate a virtual environment.

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Process an image

```bash
python app.py --image path/to/photo.jpg
```

The annotated image is saved beside the input as `photo_pose.jpg`.

To choose an output path and export landmark data:

```bash
python app.py --image path/to/photo.jpg --output output/annotated.jpg --json output/pose_data.json
```

### 2. Process a video

```bash
python app.py --video path/to/video.mp4
```

The annotated video is saved as `video_pose.mp4`.

To export pose data:

```bash
python app.py --video path/to/video.mp4 --output output/annotated.mp4 --json output/pose_data.json
```

### 3. Use a live webcam

```bash
python app.py --webcam
```

A live window opens. Press **Q** to stop.

To save the webcam session as a video and export the detected pose data:

```bash
python app.py --webcam --output output/webcam_pose.mp4 --json output/webcam_pose.json
```

### 4. Adjust detection settings

```bash
python app.py --video path/to/video.mp4 --model-complexity 2 --detection-confidence 0.6 --tracking-confidence 0.6
```

Model complexity options are `0` (fastest), `1` (balanced), and `2` (most accurate).

## Output data

The optional JSON export contains one record per detected image or video frame. Each record includes normalized landmark coordinates and visibility scores, plus joint angles in degrees. Video records also include the frame number. Frames without a detected person are omitted from the JSON export.

Example structure:

```json
[
  {
    "frame": 0,
    "landmarks": {
      "LEFT_SHOULDER": {
        "x": 0.42,
        "y": 0.25,
        "z": -0.08,
        "visibility": 0.99
      }
    },
    "joint_angles_degrees": {
      "left_elbow": 95.4,
      "right_elbow": 91.2,
      "left_shoulder": 42.1,
      "right_shoulder": 40.8,
      "left_hip": 170.3,
      "right_hip": 168.7,
      "left_knee": 175.0,
      "right_knee": 174.2
    }
  }
]
```

Values above are illustrative. Actual output depends on the input and detected pose.

## Run tests

The angle-calculation unit tests do not require a webcam:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
human-pose-estimation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── pose_estimator.py
└── tests/
    └── test_angles.py
```

## Troubleshooting

- **No person detected:** use a clear, well-lit image/video and keep the person reasonably visible in the frame.
- **Webcam unavailable:** close other applications using the camera and check operating-system camera permissions.
- **Installation issues:** use Python 3.10 or 3.11 in a fresh virtual environment, then reinstall `requirements.txt`.
- **Video playback compatibility:** the output uses the MP4V codec. If your player cannot open it, try another media player or convert the output with FFmpeg.

## Technology

- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Notes

This project is intended for learning and demonstration. Pose landmarks and calculated angles are estimates and should not be treated as clinical measurements.
