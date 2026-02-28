import cv2
import numpy as np
import math
import time
#Logitech kamera 30fps
#SQ Mini 22 23 fps



# 1. KALİBRASYON YÜKLEME
# Dosya yollarını projenin yapısına göre kontrol et
camera_matrix = np.load("../calibration/calibration_matrix.npy")
dist_coeffs = np.load("../calibration/distortion_coefficients.npy")

# 2. ÖLÇÜMLER
MARKER_LENGTH = 0.0185  # 18.5 mm
SEPARATION = 0.0055    # 5.5 mm
BOARD_SIZE = (8, 11)    # 3x3 board

# 3. DEDEKTÖR AYARLARI (Hassas Hizalama İçin)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()
# KRİTİK: Köşeleri piksellerin arasında daha hassas bulur
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 50
aruco_params.cornerRefinementMinAccuracy = 0.001
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Board tanımı
board = cv2.aruco.GridBoard(BOARD_SIZE, MARKER_LENGTH, SEPARATION, aruco_dict)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# PENCERE AYARI: WINDOW_AUTOSIZE siyah ekran sorununu genelde çözer
cv2.namedWindow("Hassas Hizalama", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Hassas Hizalama", 640, 480)

# FPS takip değişkenleri
fps = 0.0
fps_frame_count = 0
fps_time = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Görüntüyü bozmamak için kopyasını kullanmıyoruz, doğrudan üzerine çiziyoruz
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None and len(ids) > 1:
        # PnP Hesabı (Pano takibi)
        obj_points, img_points = board.matchImagePoints(corners, ids)
        retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if retval:
            # Eksenleri çiz (0.02 metre uzunluğunda)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
            
            # Koordinatları al
            x, y, z = tvec.ravel()
            rmat, _ = cv2.Rodrigues(rvec)
            gamma_deg = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))

            # Bilgileri ekrana yaz
            info = f"X:{x*100:.1f}cm Y:{y*100:.1f}cm Z:{z*100:.1f}cm Aci:{gamma_deg:.1f}"
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Terminal Hizalama: {info}")
        
        # Markerları çerçeveye al
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # FPS hesapla ve çiz
    fps_frame_count += 1
    now = time.perf_counter()
    elapsed = now - fps_time
    if elapsed >= 0.5:
        fps = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_time = now
    cv2.putText(frame, f"FPS:{fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # SADECE 'frame' değişkenini gösteriyoruz
    cv2.imshow("Hassas Hizalama", frame) += 1
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

cap.release()
cv2.destroyAllWindows()
