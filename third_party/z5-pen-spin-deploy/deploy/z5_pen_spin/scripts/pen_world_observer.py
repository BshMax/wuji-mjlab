"""Publish calibrated palm-local pen poses from an OpenCV camera."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.z5_pen_spin.lib.vision import PenObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", default="0", help="OpenCV camera index, file, or pipeline")
    parser.add_argument("--bind", default="tcp://*:5555")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    import cv2
    import zmq

    observer = PenObserver(args.config)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 1)
    capture = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    try:
        socket.bind(args.bind)
        if not capture.isOpened():
            raise RuntimeError("cannot open camera")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, observer.image_size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, observer.image_size[1])
        while True:
            # Timestamp before acquisition/detection, so processing delays count
            # toward the controller's freshness budget.
            timestamp = time.time()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame acquisition failed")
            message = observer.observe(frame, timestamp)
            socket.send_string(json.dumps(message))
            if args.preview:
                status = "VALID" if message["valid"] else "NO PALM/PEN DETECTION"
                cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Z5 pen observer", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        socket.close()
        context.term()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
