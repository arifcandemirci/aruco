import cv2
import numpy as np
from picamera2 import Picamera2
import time

#Calibration loading
camera_matrix = np.load("../calibration/calibration_matrix.npy")
dist_coeffs = np.load("../calibration/distortion_coefficients.npy")

detector = cv2.QRCodeDetector()

#Starting Camera
picam2 = Picamera2()

config =picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "YUV420"}
)

picam2.configure(config)
picam2.set_controls({"FrameDurationLimits": (16666, 16666)}) #60 Fps
picam2.start()

#cv2.namedWindow("MultiQR Pose Est.", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MultiQR Pose Est.", 320, 240)

print("[INFO] Live QR Code detection + pose started")
print("[INFO] Press 'q' to quit")

try:
    while True:
        frame = picam2.capture_array()

        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

        if retval is True:
            
            for i in range(len(decoded_info)):

                data = decoded_info[i]
                pts = points[i].astype(int)

                for j in range(4):
                     
                     cv2.line(frame,
                              tuple(pts[j]),
                              tuple(pts[(j+1) % 4]),
                              (255, 0, 0), 2)
                     
                cv2.putText(frame,
                            data,
                            tuple(pts[0]),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 255, 120),
                            2)
                
                print ("Data Found: ", data)

        cv2.imshow("QR Code Detector", frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
     print(f"[ERROR] {e}")

finally:  

    picam2.stop()
    cv2.destroyAllWindows()