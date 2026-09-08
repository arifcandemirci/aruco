from pathlib import Path

import cv2
import numpy as np

from utils import aruco_display

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
camera_matrix = np.load(CALIBRATION_DIR / "calibration_matrix.npy")
dist_coeffs = np.load(CALIBRATION_DIR / "distortion_coefficients.npy")

MARKER_LENGTH = 0.019  # meters

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow("ArUco Pose (Camera)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ArUco Pose (Camera)", 800, 600)

print("[INFO] Live ArUco detection + pose started")
print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame grab failed")
        continue

    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    output = aruco_display(corners, ids, rejected, frame)

    if ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_LENGTH,
            camera_matrix,
            dist_coeffs,
        )

        for i in range(len(ids)):
            cv2.drawFrameAxes(
                output,
                camera_matrix,
                dist_coeffs,
                rvecs[i],
                tvecs[i],
                0.009,
            )
            t = tvecs[i].ravel()
            print(
                f"ID {ids[i][0]} | x={t[0]:.3f} m "
                f"y={t[1]:.3f} m z={t[2]:.3f} m"
            )

    cv2.imshow("ArUco Pose (Camera)", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
