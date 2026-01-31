import cv2
import numpy as np
from utils import aruco_display

# Kalibrasyon yükleme
camera_matrix = np.load("../calibration/calibration_matrix.npy")
dist_coeffs = np.load("../calibration/distortion_coefficients.npy")

MARKER_LENGTH = 0.019 #metre

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

#Kamera
cap = cv2.VideoCapture(cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS ,30)

#Pencere
cv2.namedWindow("ArUco Pose (Camera)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ArUco Pose (Camera)", 800, 600)

print("[INFO] Live ArUco detection + pose started")
print("[INFO] Press 'q' to quit")

while True:
        ret, frame = cap.read()

        if not ret:
            print ("[ERROR] Frame grab failed")
            continue

        corners, ids, rejected = arucoDetector.detectMarkers(frame)

        output = aruco_display(corners, ids, rejected, frame)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_LENGTH,
                camera_matrix,
                dist_coeffs
            )

            for i in range(len(ids)):
                cv2.drawFrameAxes(
                    output,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    0.009  # eksen uzunluğu
                )
                t = tvecs[i].ravel()
                print(f"ID {ids[i][0]} | x={t[0]:.3f} m  y={t[1]:.3f} m  z={t[2]:.3f} m")
        else:
            # Marker yoksa program kapanmasın, pencere açık kalmalı
            print("[INFO] No markers detected")

        cv2.imshow("ArUco Pose (Camera)", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()