import cv2
import numpy as np
from picamera2 import Picamera2
from utils import aruco_display

#Calibration loading
camera_matrix = np.load("../calibration/calibration_matrix.npy")
dist_coeffs = np.load("../calibration/distortion_coefficients.npy")

#Aruco settings
MARKER_LENGTH = 0.019
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

#Starting camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

cv2.namedWindow("ArUco Pose (Camera)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ArUco Pose (Camera)", 800, 600)

print("[INFO] Live ArUco detection + pose started")
print("[INFO] Press 'q' to quit")

try:
    while True:
            frame_rgb = picam2.capture_array()

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

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

            fps_frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_time
            if elapsed >= 0.5:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_time = now
            cv2.putText(frame, f"FPS:{fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # SADECE 'frame' değişkenini gösteriyoruz
            cv2.imshow("Hassas Hizalama", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
     print(f"[ERROR] {e}")

finally:  

    picam2.stop()
    cv2.destroyAllWindows()