PATHGUARD - movement-based SAFE/STOP update

Replace the existing detect.py and live_webcam.py in your project with these files.
Keep camera.py, custom_bytetrack.yaml, roi_points_webcam.json, and your model weights
in the project as before.

Draw a webcam ROI if you have not already:
  python set_roi.py --source 0

Run the webcam:
  python run.py live-webcam
or:
  python live_webcam.py

Press q in the webcam window to quit.

Behavior:
  - Object outside ROI: ignored for the status.
  - Object inside ROI but stationary: SAFE.
  - Object inside ROI and moving: STOP.
  - When tracked movement ceases: SAFE immediately.

Movement is measured using the tracked object's center across a short history.
Very small motion below the configured pixel threshold may not trigger STOP.
