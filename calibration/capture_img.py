import os
import time
import cv2
from picamera2 import Picamera2

# Kayıt klasörü
IMAGE_DIR = "calibration_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Kamera başlat
picam2 = Picamera2()

# Düşük gecikme ve stabil preview için
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(0.5)

print("📸 Image capture started (IMX219 / Picamera2)")
print("COntrols:  s = save imag   q = quit")

idx = 0

while True:
    frame_rgb = picam2.capture_array()              # RGB
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("Capture (IMX219)", frame_bgr)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        path = os.path.join(IMAGE_DIR, f"img_{idx:04d}.jpg")
        cv2.imwrite(path, frame_bgr)
        print(f"✅ Saved: {path}")
        idx += 1

    elif key == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
print("🎯 Capture Finished")
