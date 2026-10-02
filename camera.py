"""
Camera / video-source helpers for Pathguard
===========================================
OpenCV-only (no YOLO / ultralytics imports) so that set_roi.py can use the very
same capture code as the live detector without pulling in the ML stack.

* open_capture()  - opens a webcam index (or a file / RTSP URL), verifies that it
                    really delivers frames, and raises CameraError with
                    troubleshooting hints if it does not.
* FrameSource     - hands frames to the main loop. For a webcam it reads in a
                    background thread and always returns the NEWEST frame, so a
                    slow YOLO step drops frames instead of building up a growing
                    lag. It also survives short camera glitches (re-opens the
                    camera) and gives up with a clear reason if the camera is gone.
"""

from __future__ import annotations

import os
import threading
import time

import cv2

# Resolution requested from webcams. The camera may pick the closest mode it
# supports - the real size is always read back from the first frame.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


class CameraError(RuntimeError):
    """The camera / video source could not be opened or stopped delivering frames."""


def parse_source(value):
    """'0' -> 0 (webcam index); anything else (file path, rtsp:// URL) stays a string."""
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def _camera_help(index):
    return (
        f"Could not get frames from camera {index}.\n"
        "  - Is another app using the webcam (Teams, Zoom, Camera app, a browser tab)? Close it and retry.\n"
        "  - Windows: Settings > Privacy & security > Camera > allow desktop apps to access the camera.\n"
        "  - A laptop privacy shutter / camera-off key may be closed.\n"
        f"  - Try another index, e.g. --source 1 (an external USB camera is often 1; current: {index}).\n"
        "  - Make sure opencv-python (not opencv-python-headless) is installed: pip install opencv-python"
    )


def open_capture(source, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, read_attempts=30):
    """Open `source` and return (cap, first_frame).

    `source` is a webcam index (int) or a file path / URL (str). For webcams the
    first reads often fail or are black while the sensor warms up, so we retry
    for a moment before giving up. Raises CameraError on failure.
    """
    if isinstance(source, int):
        # DirectShow opens much faster than the default MSMF backend on Windows
        # (MSMF can hang for ~30 s on some laptops); fall back to default if needed.
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep driver-side queue short (ignored by some backends)
            for _ in range(read_attempts):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, frame
                time.sleep(0.1)
            cap.release()
        raise CameraError(_camera_help(source))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"Could not open video source: {source}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise CameraError(f"Opened '{source}' but could not read a frame from it.")
    return cap, frame


class FrameSource:
    """Delivers frames from an already-open capture.

    threaded=True  (webcam): a reader thread keeps only the latest frame; read()
                             returns each new frame once, or None on timeout.
    threaded=False (file / stream): read() decodes the next frame synchronously so
                             no frame is skipped; None + dead=True at end of file.

    `reopen` (optional) is a zero-argument callable returning a fresh
    (cap, first_frame) - used to recover when a webcam stops delivering frames.
    """

    def __init__(self, cap, first_frame, reopen=None, threaded=True,
                 reconnect_after=2.0, give_up_after=10.0):
        self._cap = cap
        self._pending = first_frame
        self._reopen = reopen
        self._threaded = threaded
        self._reconnect_after = reconnect_after
        self._give_up_after = give_up_after

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._seq = 0
        self._seen = 0
        self._last_ok = time.monotonic()
        self._dead = False
        self._reason = ""
        self.reconnects = 0

    # -- state ---------------------------------------------------------------
    @property
    def dead(self):
        """True once the source has ended or failed for good."""
        return self._dead

    @property
    def reason(self):
        """Why the source is dead (end of file, camera lost, ...)."""
        return self._reason

    def _fail(self, reason):
        with self._cond:
            if not self._dead:
                self._dead = True
                self._reason = reason
            self._cond.notify_all()

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if not self._threaded or self._thread is not None:
            return
        with self._cond:
            self._frame, self._pending = self._pending, None
            self._seq = 1
        self._last_ok = time.monotonic()
        self._thread = threading.Thread(target=self._reader_loop, name="pathguard-camera", daemon=True)
        self._thread.start()

    def release(self):
        """Stop the reader thread and release the capture device (safe to call twice)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # -- reader thread -------------------------------------------------------
    def _reader_loop(self):
        last_reopen = 0.0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            now = time.monotonic()
            if ok and frame is not None:
                self._last_ok = now
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
                continue

            # Failed read: after a short gap try to re-open the camera.
            gap = now - self._last_ok
            if self._reopen and gap > self._reconnect_after and now - last_reopen > 1.0:
                last_reopen = now
                try:
                    self._cap.release()
                except Exception:
                    pass
                try:
                    self._cap, frame = self._reopen()
                    self.reconnects += 1
                    self._last_ok = time.monotonic()
                    with self._cond:
                        self._frame = frame
                        self._seq += 1
                        self._cond.notify_all()
                except CameraError:
                    pass  # keep trying until give_up_after expires
            time.sleep(0.01)

    # -- consumer API --------------------------------------------------------
    def _check_timeout(self):
        gap = time.monotonic() - self._last_ok
        if gap > self._give_up_after:
            self._fail(f"No frames from the camera for {gap:.0f} s - it was disconnected, "
                       "is in use by another app, or has stopped responding.")

    def read(self, timeout=0.5):
        """Return the next new frame, or None (timeout / source dead - check .dead)."""
        if not self._threaded:
            return self._read_sync()

        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == self._seen and not self._dead:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self._seq != self._seen:
                self._seen = self._seq
                return self._frame
        self._check_timeout()
        return None

    def _read_sync(self):
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        if self._dead:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._fail("End of video / stream.")
            return None
        return frame
