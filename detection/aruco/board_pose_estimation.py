from pathlib import Path
import math
import time

import cv2
import numpy as np

# Logitech camera: ~30 FPS
# SQ Mini camera: ~22-23 FPS

# 1. CALIBRATION LOADING
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
camera_matrix = np.load(CALIBRATION_DIR / "calibration_matrix.npy")
dist_coeffs = np.load(CALIBRATION_DIR / "distortion_coefficients.npy")

# 2. BOARD GEOMETRY
MARKER_LENGTH = 0.0185  # 18.5 mm
SEPARATION = 0.0055     # 5.5 mm
BOARD_SIZE = (8, 11)

# 3. DETECTOR SETTINGS
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 50
aruco_params.cornerRefinementMinAccuracy = 0.001
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

board = cv2.aruco.GridBoard(BOARD_SIZE, MARKER_LENGTH, SEPARATION, aruco_dict)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Board Pose", cv2.WINDOW_AUTOSIZE)

fps = 0.0
fps_frame_count = 0
fps_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame grab failed")
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None and len(ids) > 1:
        obj_points, img_points = board.matchImagePoints(corners, ids)

        if obj_points is not None and img_points is not None and len(obj_points) >= 4:
            retval, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
            )

            if retval:
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.01,
                )

                x, y, z = tvec.ravel()
                rmat, _ = cv2.Rodrigues(rvec)
                yaw_deg = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

                info = (
                    f"X:{x * 100:.1f}cm Y:{y * 100:.1f}cm "
                    f"Z:{z * 100:.1f}cm Yaw:{yaw_deg:.1f}deg"
                )
                cv2.putText(
                    frame,
                    info,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                print(f"[POSE] {info}")

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    fps_frame_count += 1
    now = time.perf_counter()
    elapsed = now - fps_time
    if elapsed >= 0.5:
        fps = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_time = now

    cv2.putText(
        frame,
        f"FPS:{fps:.1f}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Board Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
