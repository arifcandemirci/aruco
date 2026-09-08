from pathlib import Path
import select
import sys
import time

import cv2
import numpy as np
from picamera2 import Picamera2

IMAGE_DIR = Path(__file__).resolve().parent / "calibration_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ASCII = np.array(list(" .:-=+*#%@"))


def frame_to_ascii(frame_bgr, out_w=80):
    """Convert a BGR frame to a compact terminal ASCII preview."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    out_h = max(1, int(height * (out_w / width) * 0.5))
    small = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

    indices = (
        small.astype(np.float32) / 255.0 * (len(ASCII) - 1)
    ).astype(np.int32)
    chars = ASCII[indices]
    return "\n".join("".join(row) for row in chars)


def stdin_has_data():
    return bool(select.select([sys.stdin], [], [], 0.0)[0])


picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

print("Live preview (ASCII). Enter=save, q + Enter=quit")
index = 0
last_preview = 0.0

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        now = time.perf_counter()
        if now - last_preview > 0.10:
            last_preview = now
            preview = frame_to_ascii(frame_bgr, out_w=90)

            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(preview + "\n")
            sys.stdout.write(
                f"\n[IDX {index}] Enter=save | q+Enter=quit\n"
            )
            sys.stdout.flush()

        if stdin_has_data():
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                break

            path = IMAGE_DIR / f"img_{index:04d}.jpg"
            ok = cv2.imwrite(str(path), frame_bgr)
            sys.stdout.write(f"\nSAVED: {path} ok={ok}\n")
            sys.stdout.flush()
            index += 1

finally:
    picam2.stop()
    print("\nDone.")
