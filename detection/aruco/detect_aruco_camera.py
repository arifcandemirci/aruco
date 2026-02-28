import cv2
import argparse
from utils import ARUCO_DICT, aruco_display

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(2)

# FPS optimizasyonu için
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] Camera could not be opened")
    exit()

print("[INFO] Starting live ArUco detection...")
print("[INFO] Press 'q' to quit")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame grab failed")
        break

    corners, ids, rejected = arucoDetector.detectMarkers(frame)

    output = aruco_display(corners, ids, rejected, frame)

    cv2.namedWindow(
    "ArUco Detection (Camera)",
    cv2.WINDOW_NORMAL
    )
  
    cv2.resizeWindow(
        "ArUco Detection (Camera)",
        800, 600
    )

    cv2.imshow("ArUco Detection (Camera)", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
