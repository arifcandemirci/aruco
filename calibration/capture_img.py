import cv2
import os
import time

SAVE_DIR = "calibration_images"
TOTAL_IMAGES = 24
INTERVAL_SEC = 3

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# YUYV için EN OPTİMUM
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

print("📸 Foto çekimi başladı (YUYV optimize)")

count = 0

while count < TOTAL_IMAGES:
    start = time.time()

    # buffer temizle
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Frame alınamadı")
        break

    filename = os.path.join(SAVE_DIR, f"img_{count+1:02d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Kaydedildi: {filename}")

    count += 1

    # tam 3 sn bekle
    while time.time() - start < INTERVAL_SEC:
        time.sleep(0.01)

cap.release()
print("🎯 Tamamlandı")
