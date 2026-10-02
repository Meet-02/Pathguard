"""
Pathguard launcher
==================
One entry point for every mode. It only forwards to the existing scripts, so
`python detect.py ...` and `python set_roi.py` keep working exactly as before.

    python run.py live-webcam [options]   real-time detection on the laptop webcam
    python run.py video [options]         detect on a video file / RTSP stream (= detect.py)
    python run.py set-roi [options]       draw the ROI polygon (video, or --source 0 for the webcam)

Add --help after a command for its options, e.g.  python run.py live-webcam --help
"""

import sys

USAGE = __doc__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    # Imported lazily so `run.py --help` and `set-roi` don't load the ML stack.
    if command in ("live-webcam", "live"):
        import live_webcam
        return live_webcam.run_live(live_webcam.parse_args(rest))
    if command == "video":
        import detect
        detect.main(rest)
        return 0
    if command == "set-roi":
        import set_roi
        return set_roi.main(rest)

    print(f"Unknown command '{command}'.\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
