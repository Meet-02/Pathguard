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

Controls:
  Left-click  : Add a point
  Right-click : Remove last point
  's'         : Save polygon and exit
  'r'         : Reset all points
  'q' / ESC   : Quit without saving
"""

import json
import sys
import cv2
import numpy as np

VIDEO_PATH = "video/sample_03.mp4"
ROI_FILE = "roi_points.json"
OUTPUT_SCALE = 0.5  # must match OUTPUT_SCALE in run_on_my_video.py / detect.py

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


def main():
    global points, frame_copy, original_frame

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open '{VIDEO_PATH}'")
        sys.exit(1)

    ok, original_frame = cap.read()
    cap.release()
    if not ok:
        print("❌ Could not read first frame")
        sys.exit(1)

    # Resize to match the working resolution used by run_on_my_video.py /
    # detect.py, so the polygon you draw here lines up with what those
    # scripts actually see.
    if OUTPUT_SCALE != 1.0:
        new_w = int(original_frame.shape[1] * OUTPUT_SCALE)
        new_h = int(original_frame.shape[0] * OUTPUT_SCALE)
        original_frame = cv2.resize(original_frame, (new_w, new_h))

    print(f"Video frame size (after OUTPUT_SCALE={OUTPUT_SCALE}): "
          f"{original_frame.shape[1]}x{original_frame.shape[0]}")
    print("Draw your ROI polygon by clicking points on the frame.")
    print("Press 's' to save, 'r' to reset, 'q' to quit.\n")

    cv2.namedWindow("Draw ROI Polygon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw ROI Polygon", 1280, 720)
    cv2.setMouseCallback("Draw ROI Polygon", mouse_callback)

    frame_copy = draw_overlay(original_frame.copy(), points)
    cv2.imshow("Draw ROI Polygon", frame_copy)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(points) < 3:
                print("⚠️  Need at least 3 points to form a polygon. Keep clicking!")
                continue
            with open(ROI_FILE, 'w') as f:
                json.dump({"roi_polygon": points}, f, indent=2)
            print(f"\n✅ ROI saved to {ROI_FILE} with {len(points)} points:")
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
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()