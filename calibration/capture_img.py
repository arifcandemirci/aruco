import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import sys
import select

IMAGE_DIR = "calibration_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# ASCII karakter seti (açık->koyu)
ASCII = np.array(list(" .:-=+*#%@"))

def frame_to_ascii(frame_bgr, out_w=80):
    """BGR frame -> terminal ASCII preview string"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    # Terminal karakterleri en-boy oranı farklı: yüksekliği biraz azalt
    out_h = max(1, int(h * (out_w / w) * 0.5))
    small = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # 0..255 -> 0..len(ASCII)-1
    idx = (small.astype(np.float32) / 255.0 * (len(ASCII) - 1)).astype(np.int32)
    chars = ASCII[idx]

    # satır satır string
    lines = ["".join(row) for row in chars]
    return "\n".join(lines)

def stdin_has_data():
    return select.select([sys.stdin], [], [], 0.0)[0]

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(0.5)

print("Live preview (ASCII). Enter=save, q + Enter=quit")
idx = 0
last_preview = 0.0

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Preview'ı 10 FPS gibi bas (terminali boğmasın)
        now = time.perf_counter()
        if now - last_preview > 0.10:
            last_preview = now
            preview = frame_to_ascii(frame_bgr, out_w=90)

            # ekranı temizle + en üste yaz
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(preview + "\n")
            sys.stdout.write(f"\n[IDX {idx}] Enter=save | q+Enter=quit\n")
            sys.stdout.flush()

        # Kullanıcı input'u varsa oku (bloklamadan)
        if stdin_has_data():
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                break

            # Enter (boş) veya herhangi bir şey -> kaydet
            path = os.path.join(IMAGE_DIR, f"img_{idx:04d}.jpg")
            ok = cv2.imwrite(path, frame_bgr)
            sys.stdout.write(f"\nSAVED: {path} ok={ok}\n")
            sys.stdout.flush()
            idx += 1

finally:
    picam2.stop()
    print("\nDone.")
