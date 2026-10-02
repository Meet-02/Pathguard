# Pathguard — Blind Corner Alert

YOLO11 + ByteTrack detection of people and vehicles inside a configurable Region of Interest (ROI).
While something is inside the ROI the status is **STOP**, otherwise **SAFE**. On the full system STOP
is sent to an Arduino Nano (relay); on a laptop it is simply shown on screen and logged.

```
Laptop webcam → YOLO object detection → Object tracking + ROI checking → SAFE / STOP
                                                                   (existing safety logic)
```

Detected classes (COCO): person, bicycle, car, motorcycle (also scooters), bus, truck.

## Modes

| Command | What it does |
|---|---|
| `python run.py live-webcam` | **Real-time detection on the laptop webcam** (new) |
| `python run.py set-roi --source 0` | Draw the ROI on a webcam frame (new option of `set_roi.py`) |
| `python detect.py --source clip.mp4 --out out.mp4` | Detect on a video file / RTSP stream (unchanged) |
| `python run_on_my_video.py` | Edit-the-CONFIG-block video runner (unchanged) |
| `python set_roi.py` | Draw the ROI on a video's first frame (unchanged default) |

`python run.py video ...` and `python run.py set-roi ...` are just shortcuts to `detect.py` / `set_roi.py`;
the original commands keep working exactly as before.

## Requirements

* Python 3.9 – 3.12 (64-bit)
* Windows 10/11, macOS or Linux, with a webcam for live mode
* Packages: `ultralytics` (brings PyTorch), `opencv-python`, `lapx` — see `requirements.txt`
  (`pyserial` only if you connect the Arduino)
* Internet on the **first run**: the `yolo11n.pt` weights are downloaded automatically.
  No weights are stored in this repo.

## Install (Windows + VS Code)

1. Open the `Pathguard-main` folder in VS Code (**File → Open Folder**).
2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:
   ```powershell
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```
   If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
3. Install the packages:
   ```powershell
   pip install -r requirements.txt
   ```
4. In VS Code press **Ctrl+Shift+P → Python: Select Interpreter** and pick the `venv` one.

(macOS/Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.)

## Run live webcam detection

```powershell
python run.py live-webcam
```

A resizable window opens showing the camera with boxes, class, confidence and tracking ID per object,
the ROI outline, FPS, object counts and the **SAFE / STOP** banner. **Press `q` to quit** (Ctrl+C in the terminal
also works, and closing the window with ✕ is detected too). The camera is always released on exit.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--source N` | `0` | Camera index. Use `1` for an external USB camera. |
| `--confidence X` | `0.25` | Detection threshold, 0–1 (same default as `detect.py`). Higher = fewer false detections, more misses. |
| `--roi FILE` | `roi_points_webcam.json` | ROI file (see below). |
| `--save-video` | off | Record the annotated view. |
| `--output-dir DIR` | `output/webcam_recordings` | Where recordings go. |
| `--width` / `--height` | `1280` / `720` | Requested camera resolution (the camera picks the nearest mode; `0` = camera default). |
| `--imgsz N` | model default (640) | Smaller (e.g. `320`) = faster on a slow CPU, less accurate for small/far objects. |
| `--device D` | auto | `cpu`, or `0` for the first NVIDIA GPU. |
| `--model FILE` | `yolo11n.pt` | Other YOLO weights. |
| `--use-relay`, `--relay-port` | off, `COM3` (Windows) | Real Arduino output — see *Hardware*. |
| `--no-display`, `--max-frames N` | off | Headless run / stop after N frames (useful for tests). |

Examples:

```powershell
python run.py live-webcam --source 1                      # second camera
python run.py live-webcam --confidence 0.5 --save-video   # stricter threshold + recording
python run.py live-webcam --imgsz 320                     # more FPS on a slow laptop
```

### Recording

`--save-video` writes `output/webcam_recordings/webcam_YYYYMMDD_HHMMSS.mp4` (mp4v codec) containing
exactly what you see (boxes, ROI, banner, FPS). The file uses the camera's real frame size and the
**measured** processing FPS (measured over the first 30 frames, so playback speed is correct even if
the detector is slower than the camera). Nothing is recorded unless you pass `--save-video`; the
`output/` folder is already git-ignored.

## Configure the ROI

Only objects whose **box centre is inside the ROI polygon** count towards STOP (existing rule).
Objects elsewhere are drawn grey with "outside ROI".

The ROI in `roi_points.json` was drawn on a *recorded video*, so it does not describe your laptop's
view. The webcam therefore has its own file:

```powershell
python set_roi.py --source 0        # or: python run.py set-roi --source 0
```

A live preview opens → aim the camera → **SPACE** freezes a frame → left-click the polygon corners
(right-click = undo, `r` = reset) → **`s`** saves to `roi_points_webcam.json` → `q` quits without saving.
The file also records the frame size it was drawn on, so if the camera later runs at another
resolution the polygon is scaled automatically. Your video ROI (`roi_points.json`) is never touched.

**If there is no ROI** (file missing, invalid, or a video-ROI whose points don't fit the camera
frame) the program says so in the console and on the video (`NO ROI - whole frame counts`) and
falls back to the existing `detect.py` behaviour: *any* person/vehicle anywhere in the frame gives
STOP. It never silently guesses a region. To use another file: `--roi path\to\file.json`.

Behaviour kept from the existing logic: STOP is held for 15 frames after the last object leaves the
ROI (debounce), and a console line `[ALERT] STOP …` is printed once per entry — not once per frame.

Tracking and thresholds (`custom_bytetrack.yaml`, `ALERT_HOLD_FRAMES`, `MOVEMENT_THRESHOLD_PX`, …)
are unchanged and live in the same files as before.

## Running without the Raspberry Pi / Arduino

By default live mode is **software-only**: SAFE/STOP is shown on screen and `[Relay] -> STOP` is
printed; nothing is sent over serial. Nothing needs to be connected.

With hardware: `pip install pyserial`, then
`python run.py live-webcam --use-relay --relay-port COM5` (Windows) or `--relay-port /dev/ttyACM0` (Pi).
Relay writes happen in a background thread, so a slow or unplugged serial port can't freeze the video.
Unlike the video mode (which writes on every frame), live mode sends a command when the state
**changes** and repeats it once per second as a keep-alive. If your Arduino sketch needs a faster
heartbeat, change `RELAY_KEEPALIVE_S` in `live_webcam.py`. On exit the relay is left in its last state.

## Existing video mode (unchanged)

```powershell
python detect.py --source clip.mp4 --out annotated.mp4 --show
python run_on_my_video.py        # edit VIDEO_PATH etc. at the top of the file
```
`detect.py` now runs its per-frame logic through a shared `SafetyPipeline` class (the same one live
mode uses). Its output was verified to be pixel-identical to the previous version. The tracker config
is now also found when you start the script from a different folder.

## Troubleshooting

| Problem | Try |
|---|---|
| `Could not get frames from camera 0` | Close Teams/Zoom/Camera app/browser tabs using the camera; check Windows **Settings → Privacy & security → Camera** (allow desktop apps); open the privacy shutter; try `--source 1`. |
| Window never opens / OpenCV GUI error | Make sure `opencv-python` (not `-headless`) is installed: `pip uninstall opencv-python-headless` then `pip install opencv-python`. |
| Low FPS | `--imgsz 320`; close other apps; on an NVIDIA GPU install the CUDA build of PyTorch and use `--device 0`. Frames are dropped rather than queued, so the view stays real-time. |
| Picture lags or freezes briefly | The camera is re-opened automatically after ~2 s without frames; after 10 s the program stops with an error message. |
| `Could not load model` | First run needs internet to download `yolo11n.pt`. |
| `No module named lap` | `pip install lapx` |
| Too many false STOPs | Raise `--confidence` (e.g. 0.5) and/or draw a tighter ROI. |
| Person not detected | Lower `--confidence`, improve lighting, make sure they are in the ROI. |

## Limitations

* Pretrained COCO classes only: auto-rickshaws have no class and are detected inconsistently
  (see the note in `detect.py`).
* A laptop webcam is not the CP Plus corner camera: its view, ROI and thresholds will differ.
* Live mode is as safe as its detector — a missed detection is a missed STOP. If detection fails for a
  frame the previous status is held; after 30 failed frames in a row the program stops with an error.
* Relay/Arduino behaviour is **not** exercised by laptop testing.

## Tests

```powershell
python -m unittest discover -s tests -v
```
Uses a scripted stand-in for YOLO and simulated cameras (no webcam, GPU or download needed). It covers
SAFE/STOP + ROI logic, alert de-duplication, ROI loading/scaling, camera failure/recovery, recording,
`q`/Ctrl+C shutdown, and error paths. It does **not** measure YOLO accuracy, real-webcam behaviour, or
hardware — check those on your machine (steps above).

## Files

```
run.py                launcher: live-webcam | video | set-roi
live_webcam.py        live webcam mode (HUD, recording, alerts, CLI)      [new]
camera.py             camera opening + threaded latest-frame reader       [new]
detect.py             YOLO11 + ByteTrack + ROI + SAFE/STOP (SafetyPipeline extracted; CLI unchanged)
set_roi.py            ROI drawing (video, or --source 0 for webcam)
run_on_my_video.py    video runner (unchanged)
custom_bytetrack.yaml tracker settings (unchanged)
roi_points.json       ROI for the recorded video (unchanged)
roi_points_webcam.json  created by `set_roi.py --source 0`
tests/                unit + integration tests                            [new]
```
