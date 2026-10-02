"""
ROI (Region of Interest) Polygon Drawer
========================================
Opens the first frame of your video and lets you draw a polygon
by clicking points. The polygon is saved to roi_points.json.

IMPORTANT: OUTPUT_SCALE below MUST match the OUTPUT_SCALE used in
run_on_my_video.py / detect.py. Those scripts resize every frame by that
factor before doing anything else — if you draw your ROI on the full-res
frame here but the other script is working on a resized frame, the polygon
will not line up with the objects you actually see in the output video.

WEBCAM: draw the ROI on your laptop camera instead of a video file:
    python set_roi.py --source 0          (or: python run.py set-roi --source 0)
  A live preview opens - frame the scene, press SPACE to freeze it, then click
  your polygon. It is saved to roi_points_webcam.json (so your video ROI in
  roi_points.json is NOT overwritten) together with the frame size it was drawn
  on, which lets live_webcam.py scale it if the camera resolution differs.

Other options:  --roi-file PATH   --scale FACTOR   --width/--height (camera)

Controls:
  Left-click  : Add a point
  Right-click : Remove last point
  's'         : Save polygon and exit
  'r'         : Reset all points
  'q' / ESC   : Quit without saving
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

from camera import (DEFAULT_HEIGHT, DEFAULT_WIDTH, CameraError, open_capture,
                    parse_source)

VIDEO_PATH = "video/sample_03.mp4"
ROI_FILE = "roi_points.json"
OUTPUT_SCALE = 0.5  # must match OUTPUT_SCALE in run_on_my_video.py / detect.py
WEBCAM_ROI_FILE = "roi_points_webcam.json"  # used by default when --source is a camera index

points = []
frame_copy = None
original_frame = None


def draw_overlay(img, pts):
    overlay = img.copy()
    if len(pts) > 0:
        # draw filled semi-transparent polygon
        if len(pts) > 2:
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (255, 0, 255, 80))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [poly], isClosed=True, color=(255, 0, 255), thickness=2)
        elif len(pts) == 2:
            cv2.line(img, pts[0], pts[1], (255, 0, 255), 2)

        # draw each vertex
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 255, 255), -1)
            cv2.circle(img, p, 6, (255, 0, 255), 2)
            cv2.putText(img, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # instructions
    cv2.putText(img, "Left-click: add point | Right-click: undo | 's': save | 'r': reset | 'q': quit",
                (10, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, f"Points: {len(pts)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def mouse_callback(event, x, y, flags, param):
    global points, frame_copy, original_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        frame_copy = draw_overlay(original_frame.copy(), points)
        cv2.imshow("Draw ROI Polygon", frame_copy)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            frame_copy = draw_overlay(original_frame.copy(), points)
            cv2.imshow("Draw ROI Polygon", frame_copy)


def grab_reference_frame(source, width, height):
    """Return one frame to draw on. Video file: its first frame. Webcam: live preview, SPACE freezes a frame."""
    if not isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Could not open '{source}'")
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("❌ Could not read first frame")
            return None
        return frame

    try:
        cap, frame = open_capture(source, width, height)
    except CameraError as exc:
        print(f"❌ {exc}")
        return None
    print("Webcam preview open - frame the scene, press SPACE to freeze a frame, q/ESC to cancel.")
    title = "Webcam preview - SPACE: freeze frame | q: cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    chosen = None
    try:
        while True:
            ok, live = cap.read()
            if ok and live is not None:
                frame = live
            shown = frame.copy()
            cv2.putText(shown, "SPACE: freeze this frame and draw ROI   |   q: cancel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(title, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                chosen = frame.copy()
                break
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyWindow(title)
    return chosen


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Draw the ROI polygon on a video's first frame or on a webcam frame.")
    p.add_argument("--source", type=parse_source, default=VIDEO_PATH,
                   help=f"video file path (default {VIDEO_PATH}) or webcam index such as 0")
    p.add_argument("--roi-file", default=None,
                   help=f"where to save the polygon (default {ROI_FILE} for a video, {WEBCAM_ROI_FILE} for a webcam)")
    p.add_argument("--scale", type=float, default=None,
                   help=f"resize factor (default {OUTPUT_SCALE} for a video - must match detect.py; 1.0 for a webcam)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="requested webcam width (webcam only)")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="requested webcam height (webcam only)")
    return p.parse_args(argv)


def main(argv=None):
    global points, frame_copy, original_frame
    args = parse_args(argv)
    is_webcam = isinstance(args.source, int)
    roi_file = args.roi_file or (WEBCAM_ROI_FILE if is_webcam else ROI_FILE)
    scale = args.scale if args.scale is not None else (1.0 if is_webcam else OUTPUT_SCALE)

    original_frame = grab_reference_frame(args.source, args.width, args.height)
    if original_frame is None:
        return 1
    points = []

    # Resize to match the working resolution used by run_on_my_video.py /
    # detect.py, so the polygon you draw here lines up with what those
    # scripts actually see.
    if scale != 1.0:
        new_w = int(original_frame.shape[1] * scale)
        new_h = int(original_frame.shape[0] * scale)
        original_frame = cv2.resize(original_frame, (new_w, new_h))

    print(f"Frame size (after scale={scale}): "
          f"{original_frame.shape[1]}x{original_frame.shape[0]}")
    print("Draw your ROI polygon by clicking points on the frame.")
    print("Press 's' to save, 'r' to reset, 'q' to quit.\n")

    cv2.namedWindow("Draw ROI Polygon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw ROI Polygon", 1280, 720)
    cv2.setMouseCallback("Draw ROI Polygon", mouse_callback)

    frame_copy = draw_overlay(original_frame.copy(), points)
    cv2.imshow("Draw ROI Polygon", frame_copy)

    result = 0
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(points) < 3:
                print("⚠️  Need at least 3 points to form a polygon. Keep clicking!")
                continue
            # "frame_size" is new: it lets live_webcam.py rescale the polygon if the camera
            # runs at a different resolution later. Older readers only use "roi_polygon".
            with open(roi_file, 'w') as f:
                json.dump({"roi_polygon": points,
                           "frame_size": [original_frame.shape[1], original_frame.shape[0]]}, f, indent=2)
            print(f"\n✅ ROI saved to {roi_file} with {len(points)} points:")
            for i, p in enumerate(points):
                print(f"   Point {i+1}: ({p[0]}, {p[1]})")
            break
        elif key == ord('r'):
            points = []
            frame_copy = draw_overlay(original_frame.copy(), points)
            cv2.imshow("Draw ROI Polygon", frame_copy)
            print("🔄 Points reset")
        elif key == ord('q') or key == 27:
            print("❌ Quit without saving")
            result = 1
            break

    cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    sys.exit(main())
