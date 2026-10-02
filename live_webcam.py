"""
Pathguard - live webcam detection
=================================
Runs the SAME YOLO11 + ByteTrack + ROI + SAFE/STOP pipeline as detect.py
(detect.SafetyPipeline) on frames from a laptop webcam, and shows the result in
a live window.

    python run.py live-webcam                       # default webcam (index 0)
    python run.py live-webcam --source 1            # another camera
    python run.py live-webcam --confidence 0.5 --save-video
    python live_webcam.py --help                    # same thing, without run.py

Keys:  q = quit (releases the camera, finishes any recording, closes the window)

Without Raspberry Pi hardware this runs in software-only mode: the Arduino/relay
is simulated (a "[Relay] -> STOP" line is printed). Pass --use-relay (and
--relay-port COMx) only when the Arduino is actually connected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from camera import (DEFAULT_HEIGHT, DEFAULT_WIDTH, CameraError, FrameSource,
                    open_capture, parse_source)
from detect import (CONF_THRESHOLD, YOLO, RelayController, SafetyPipeline,
                    SafetyState)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The ROI in roi_points.json was drawn on a recorded VIDEO's frame. A webcam sees a
# different scene (and size), so the webcam gets its own file - create it with:
#   python set_roi.py --source 0
DEFAULT_ROI_FILE = os.path.join(BASE_DIR, "roi_points_webcam.json")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "webcam_recordings")
WINDOW_NAME = "Pathguard - Live Webcam (q = quit)"

RELAY_KEEPALIVE_S = 1.0               # re-send the current state this often (the old file/video mode re-sent every frame)
MAX_CONSECUTIVE_INFERENCE_ERRORS = 30  # give up after this many failed frames in a row
RECORD_FPS_WARMUP_FRAMES = 30         # frames used to measure the real processing FPS before opening the video file

EXIT_OK, EXIT_CAMERA, EXIT_MODEL, EXIT_RUNTIME = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# ROI
# ----------------------------------------------------------------------------
def load_live_roi(roi_file, frame_w, frame_h):
    """Load the ROI polygon for the live camera frame.

    Returns (polygon | None, status_text). status_text is shown on the video.

    * If the file records the `frame_size` it was drawn on (set_roi.py does this),
      the polygon is scaled to the actual camera resolution.
    * If there is no ROI (file missing / invalid / does not fit the frame) we do
      NOT guess: we say so loudly and fall back to the existing detect.py
      behaviour - every alert-class object anywhere in the frame counts. That is
      the fail-safe direction (more STOPs, never fewer).
    """
    no_roi = "NO ROI - whole frame counts"
    if not roi_file or not os.path.exists(roi_file):
        print(f"[ROI] WARNING: no ROI file found ({roi_file}).\n"
              "      Running WITHOUT an ROI: any person/vehicle anywhere in the frame triggers STOP.\n"
              "      Draw one with:  python set_roi.py --source 0")
        return None, no_roi

    try:
        with open(roi_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        points = np.array(data.get("roi_polygon", []), dtype=np.float64)
    except (OSError, ValueError) as exc:
        print(f"[ROI] WARNING: could not read {roi_file} ({exc}) - running WITHOUT an ROI.")
        return None, no_roi

    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        print(f"[ROI] WARNING: {roi_file} needs at least 3 [x, y] points - running WITHOUT an ROI.")
        return None, no_roi

    ref = data.get("frame_size")
    if isinstance(ref, (list, tuple)) and len(ref) == 2 and ref[0] > 0 and ref[1] > 0:
        sx, sy = frame_w / float(ref[0]), frame_h / float(ref[1])
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            print(f"[ROI] drawn at {ref[0]}x{ref[1]}, camera is {frame_w}x{frame_h} - scaling polygon.")
        points = points * np.array([sx, sy])
    else:
        # Legacy file (e.g. roi_points.json) - we cannot know which frame size it was drawn for.
        if (points[:, 0].min() < 0 or points[:, 1].min() < 0
                or points[:, 0].max() > frame_w or points[:, 1].max() > frame_h):
            print(f"[ROI] WARNING: {roi_file} has points outside the {frame_w}x{frame_h} camera frame and "
                  "records no frame size - it was drawn for a different video. Running WITHOUT an ROI.\n"
                  "      Draw a new one with:  python set_roi.py --source 0")
            return None, no_roi
        print(f"[ROI] WARNING: {roi_file} records no frame size; assuming it was drawn on a "
              f"{frame_w}x{frame_h} frame. Re-draw it with set_roi.py --source 0 to be sure.")

    polygon = np.round(points).astype(np.int32)
    print(f"[ROI] polygon loaded from {roi_file} ({len(polygon)} vertices)")
    return polygon, f"ROI: {len(polygon)} points"


# ----------------------------------------------------------------------------
# Alerts (relay) - never block the video loop
# ----------------------------------------------------------------------------
class AlertDispatcher:
    """Sends SAFE/STOP to the relay controller from a background thread.

    The video loop only calls update(state) (instant). The worker sends the state
    when it CHANGES, and re-sends it every `keepalive` seconds - so an object that
    sits in the ROI does not cause a flood of repeated alerts, and a slow or stuck
    serial port can never freeze the camera window.
    """

    def __init__(self, relay, keepalive=RELAY_KEEPALIVE_S):
        self._relay = relay
        self._keepalive = keepalive
        self._latest = SafetyState.SAFE
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pathguard-alerts", daemon=True)

    def start(self):
        self._thread.start()

    def update(self, state):
        if state != self._latest:
            self._latest = state
            self._wake.set()

    def _loop(self):
        last_sent, last_t = None, 0.0
        while not self._stop.is_set():
            self._wake.clear()          # clear BEFORE reading state so an update can't be missed
            state = self._latest
            if state != last_sent or time.monotonic() - last_t >= self._keepalive:
                try:
                    self._relay.set_state(state)
                except Exception as exc:  # e.g. serial cable pulled - keep the video running
                    print(f"\n[Relay] write failed: {exc}")
                last_sent, last_t = state, time.monotonic()
            self._wake.wait(timeout=self._keepalive)

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------
class RecordingWriter:
    """Saves the annotated live view to output/webcam_recordings/webcam_<timestamp>.mp4.

    A webcam's reported FPS is often not what the detector really achieves, and a
    file written at the wrong FPS plays back too fast/slow. So the first
    RECORD_FPS_WARMUP_FRAMES frames are held in memory while the real FPS is
    measured, then the file is opened with the measured FPS and the actual frame size.
    """

    def __init__(self, out_dir, frame_size, fallback_fps=20.0, warmup=RECORD_FPS_WARMUP_FRAMES):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, datetime.now().strftime("webcam_%Y%m%d_%H%M%S.mp4"))
        self.frame_size = frame_size  # (w, h)
        self.fps = None
        self.failed = False
        self._fallback_fps = fallback_fps if fallback_fps and fallback_fps > 0 else 20.0
        self._warmup = warmup
        self._buffer, self._times = [], []
        self._writer = None

    def write(self, frame):
        if self.failed:
            return
        if self._writer is None:
            self._buffer.append(frame)
            self._times.append(time.monotonic())
            if len(self._buffer) >= self._warmup:
                self._open_and_flush()
        else:
            self._write_one(frame)

    def _write_one(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)

    def _open_and_flush(self):
        n = len(self._times)
        if n >= 2 and self._times[-1] > self._times[0]:
            fps = (n - 1) / (self._times[-1] - self._times[0])
        else:
            fps = self._fallback_fps
        self.fps = float(min(max(fps, 1.0), 60.0))
        self._writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size)
        if not self._writer.isOpened():
            print(f"[REC] WARNING: could not create {self.path} - recording disabled.")
            self.failed, self._writer = True, None
        else:
            for frame in self._buffer:
                self._write_one(frame)
        self._buffer, self._times = [], []

    def release(self):
        if self._writer is None and self._buffer and not self.failed:
            self._open_and_flush()
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_roi_overlay(frame, polygon, state):
    """Translucent ROI fill + outline: green when SAFE, red when STOP (same look as run_on_my_video.py)."""
    fill, border = ((0, 180, 0), (0, 255, 0)) if state == SafetyState.SAFE else ((0, 0, 200), (0, 0, 255))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], fill)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=border, thickness=3)
    return frame


def _put_text(frame, text, org, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, state, result, fps, roi_status, roi_active, recording=False, notice=""):
    """Status banner + FPS + counts + tracking IDs (+ REC dot and any warning notice)."""
    h, w = frame.shape[:2]
    stop = state == SafetyState.STOP
    cv2.rectangle(frame, (0, 0), (360, 50), (0, 0, 220) if stop else (0, 160, 0), -1)
    cv2.putText(frame, f"STATUS: {state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    detected = result.num_detected if result else 0
    in_roi = result.num_in_roi if result else 0
    ids = result.roi_track_ids if result else []
    _put_text(frame, f"FPS: {fps:5.1f}", (10, 75))
    _put_text(frame, f"Detected: {detected}   In ROI: {in_roi}", (10, 98))
    _put_text(frame, "IDs in ROI: " + (", ".join(str(i) for i in ids) if ids else "-"), (10, 121))
    _put_text(frame, roi_status, (10, 144), (0, 220, 0) if roi_active else (0, 165, 255))

    if recording:
        cv2.circle(frame, (w - 70, 25), 9, (0, 0, 255), -1)
        _put_text(frame, "REC", (w - 55, 31), (255, 255, 255), 0.6, 2)
    if notice:
        _put_text(frame, notice, (10, h - 15), (0, 165, 255), 0.7, 2)
    return frame


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def _quit_requested(delay_ms=1):
    """True if q was pressed or the window was closed with the X button."""
    key = cv2.waitKey(delay_ms) & 0xFF
    if key in (ord("q"), ord("Q")):
        return True
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run_live(args):
    """Run live detection. Returns a process exit code (EXIT_*)."""
    source = args.source
    is_webcam = isinstance(source, int)
    show = not args.no_display

    # 1) Camera first, so a missing/busy camera is reported immediately.
    label = f"camera {source}" if is_webcam else str(source)
    print(f"Opening {label} ...")
    try:
        cap, first_frame = open_capture(source, args.width, args.height)
    except CameraError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return EXIT_CAMERA

    frame_h, frame_w = first_frame.shape[:2]
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"{label}: {frame_w}x{frame_h}" + (f", driver reports {cam_fps:.0f} FPS" if cam_fps else ""))

    reopen = (lambda: open_capture(source, args.width, args.height)) if is_webcam else None
    frames = FrameSource(cap, first_frame, reopen=reopen, threaded=is_webcam)
    dispatcher = recorder = None
    exit_code = EXIT_OK
    window_open = False
    n_alerts = n_frames = 0
    t_start = time.monotonic()

    try:
        # 2) Model (weights download on the very first run).
        print(f"Loading YOLO model ({args.model}) ...")
        try:
            model = YOLO(args.model)
        except Exception as exc:
            print(f"\n[ERROR] Could not load model '{args.model}': {exc}\n"
                  "        The first run downloads the weights, so it needs an internet connection.",
                  file=sys.stderr)
            return EXIT_MODEL

        roi_polygon, roi_status = load_live_roi(args.roi, frame_w, frame_h)
        pipeline = SafetyPipeline(model, roi_polygon, conf=args.confidence,
                                  imgsz=args.imgsz, device=args.device)

        relay = RelayController(port=args.relay_port, enabled=args.use_relay)
        if not args.use_relay:
            print("[Relay] software-only mode (no hardware) - SAFE/STOP is shown and logged only.")
        dispatcher = AlertDispatcher(relay)
        dispatcher.start()

        if args.save_video:
            recorder = RecordingWriter(args.output_dir, (frame_w, frame_h), fallback_fps=cam_fps)
            print(f"[REC] recording annotated video to {recorder.path}")

        if show:
            try:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, min(frame_w, 1280), int(min(frame_w, 1280) * frame_h / frame_w))
                window_open = True
            except cv2.error as exc:
                print(f"\n[ERROR] OpenCV cannot open a window ({exc}).\n"
                      "        If opencv-python-headless is installed, run: pip uninstall opencv-python-headless "
                      "&& pip install opencv-python\n        Or run with --no-display.", file=sys.stderr)
                return EXIT_RUNTIME

        frames.start()
        print("Running - press q in the video window to quit." if show else "Running headless - press Ctrl+C to stop.")

        fps, stamps = 0.0, deque(maxlen=30)   # FPS = frames in the last ~30 / their time span
        consecutive_errors = 0
        result, last_shown, notice = None, None, ""
        logged_state = SafetyState.SAFE

        while True:
            frame = frames.read(timeout=0.5)

            if frame is None:
                if frames.dead:
                    if is_webcam:
                        print(f"\n[ERROR] {frames.reason}", file=sys.stderr)
                        exit_code = EXIT_CAMERA
                    else:
                        print(f"\n{frames.reason}")
                    break
                # Temporary stall: keep the window responsive and tell the user.
                if show:
                    if last_shown is not None:
                        idle = last_shown.copy()
                        _put_text(idle, "CAMERA: no frames - retrying...", (10, idle.shape[0] - 15),
                                  (0, 165, 255), 0.7, 2)
                        cv2.imshow(WINDOW_NAME, idle)
                    if _quit_requested(30):
                        break
                continue

            n_frames += 1

            # --- detection + tracking + ROI + SAFE/STOP (shared with detect.py) ---
            try:
                result = pipeline.process(frame)
                consecutive_errors = 0
                notice = ""
            except Exception as exc:  # inference hiccup - hold the previous status, don't crash
                consecutive_errors += 1
                notice = "DETECTION ERROR - holding last status"
                if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                    print(f"\n[WARN] detection failed on a frame ({exc.__class__.__name__}: {exc}) "
                          f"[{consecutive_errors} in a row]")
                if consecutive_errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
                    print(f"\n[ERROR] detection failed on {consecutive_errors} consecutive frames - stopping.",
                          file=sys.stderr)
                    exit_code = EXIT_RUNTIME
                    break

            state = pipeline.state
            dispatcher.update(state)
            if result is not None and result.new_alert:
                n_alerts += 1
            if state != logged_state:  # one console line per transition, not per frame
                ids = result.roi_track_ids if result else []
                print(f"\n[ALERT] {'STOP - object in ROI, ids ' + str(ids) if state == SafetyState.STOP else 'cleared - SAFE'}"
                      f" (frame {pipeline.frame_idx})")
                logged_state = state

            stamps.append(time.monotonic())
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])

            if roi_polygon is not None:
                draw_roi_overlay(frame, roi_polygon, state)
            draw_hud(frame, state, result, fps, roi_status, roi_polygon is not None,
                     recording=recorder is not None and not recorder.failed, notice=notice)

            if recorder is not None:
                recorder.write(frame)
            if show:
                cv2.imshow(WINDOW_NAME, frame)
                last_shown = frame
                if _quit_requested(1):
                    break
            if args.max_frames and n_frames >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down.")
    finally:
        # Always runs: normal quit, q, Ctrl+C, or any exception.
        frames.release()
        if dispatcher is not None:
            dispatcher.stop()
        if recorder is not None:
            recorder.release()
            if not recorder.failed and recorder.fps:
                print(f"[REC] saved {recorder.path} ({recorder.fps:.1f} FPS)")
        if window_open:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = max(time.monotonic() - t_start, 1e-6)
    print(f"\nProcessed {n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} FPS), "
          f"{n_alerts} alert(s). Camera released.")
    return exit_code


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser(prog="python run.py live-webcam"):
    default_relay_port = "COM3" if os.name == "nt" else "/dev/ttyACM0"
    p = argparse.ArgumentParser(prog=prog, description="Pathguard - real-time webcam detection (YOLO11 + ByteTrack + ROI + SAFE/STOP)")
    p.add_argument("--source", type=parse_source, default=0,
                   help="camera index (default 0; try 1 for an external camera). A video file path / URL also works for testing.")
    p.add_argument("--confidence", type=float, default=CONF_THRESHOLD,
                   help=f"detection confidence threshold, 0-1 (default {CONF_THRESHOLD}, same as detect.py)")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (default yolo11n.pt, downloaded on first run)")
    p.add_argument("--roi", default=DEFAULT_ROI_FILE,
                   help="ROI JSON file (default roi_points_webcam.json, create it with: python set_roi.py --source 0)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"requested camera width (default {DEFAULT_WIDTH}; 0 = camera default)")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"requested camera height (default {DEFAULT_HEIGHT}; 0 = camera default)")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO inference size, e.g. 320 for more FPS on a slow CPU (default: model default)")
    p.add_argument("--device", default=None, help="inference device, e.g. cpu or 0 for the first GPU (default: auto)")
    p.add_argument("--save-video", action="store_true", help="record the annotated live view (off by default)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="folder for --save-video recordings (default output/webcam_recordings)")
    p.add_argument("--use-relay", action="store_true", help="really send SAFE/STOP to the Arduino over serial (default: simulated)")
    p.add_argument("--relay-port", default=default_relay_port, help=f"serial port of the Arduino (default {default_relay_port})")
    p.add_argument("--no-display", action="store_true", help="headless: no window (stop with Ctrl+C or --max-frames)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = run until q)")
    return p


def parse_args(argv=None, prog="python run.py live-webcam"):
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.width < 0 or args.height < 0:
        parser.error("--width/--height must not be negative")
    return args


def main(argv=None):
    return run_live(parse_args(argv, prog="python live_webcam.py"))


if __name__ == "__main__":
    sys.exit(main())
