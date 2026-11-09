# Human Pose Estimation

A browser-based human pose estimation demo, deployable as a static site on **Vercel**. It uses MediaPipe Pose Landmarker (Tasks Vision) to detect body landmarks from a live camera or uploaded image, draw a connected skeleton, and calculate selected joint angles.

The original Python/OpenCV implementation is retained in `app.py` and `src/` for command-line image/video processing. The Vercel-hosted browser experience is served from the repository root.

## Browser features

- Live camera pose tracking (33 MediaPipe landmarks).
- Connected skeleton overlay.
- Estimated left/right elbow, shoulder, hip, and knee angles.
- Single-image analysis.
- Browser-local processing: camera frames are not uploaded by this app.

## Deploy to Vercel from GitHub

1. Push this repository to GitHub.
2. Sign in to Vercel and choose **Add New → Project**.
3. Import the `human-pose-estimation` GitHub repository.
4. Set **Framework Preset** to `Other`.
5. Leave **Root Directory** as `./`.
6. Leave Build Command and Output Directory blank (static site).
7. Deploy.

Vercel will redeploy automatically when you push changes to the connected GitHub branch.

## Run locally

The web app must be served over HTTP/HTTPS; opening `index.html` directly as a file may prevent camera access.

```bash
npx serve .
```

Open the local URL printed by the server. Camera access requires HTTPS or localhost and browser permission.

## Python implementation

The Python version remains available for local processing:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python app.py --webcam
```

Image and video examples:

```bash
python app.py --image path/to/photo.jpg
python app.py --video path/to/video.mp4 --json output/pose.json
python -m unittest discover -s tests -v
```

## Important notes

- The browser version downloads the MediaPipe JavaScript runtime, WebAssembly files, and pose model from public CDNs at runtime; an internet connection is required.
- Camera access requires HTTPS (Vercel provides HTTPS) and explicit browser permission.
- GPU delegate support varies by device/browser. If a device reports a GPU initialization error, change `delegate: "GPU"` to `delegate: "CPU"` in `app.js`.
- Pose landmarks and angles are estimates, not clinical measurements.
- The Python/OpenCV webcam and video-processing features run locally; Vercel static hosting does not run that persistent camera-processing loop on the server.
