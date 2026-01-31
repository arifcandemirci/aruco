import cv2
import numpy as np
import os

# ====== 1. AYARLAR VE PARAMETRELER ======
# Önemli: charuco_real.png görseline göre satır sayısını 8 olarak güncelledim.
BOARD_SIZE = (5, 7) 
CHARUCO_SQUARE_SIZE = 0.039   # metre (39mm)
CHARUCO_MARKER_SIZE = 0.019   # metre (19mm)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250 # Önceki konuşmamıza göre 250'lik sözlük

IMAGE_DIR = "calibration_images" 
SAVE_DIR = "."

# ====== 2. BOARD VE DEDEKTÖR TANIMLAMA ======
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    CHARUCO_SQUARE_SIZE,
    CHARUCO_MARKER_SIZE,
    aruco_dict
)

# OpenCV 4.12.0 için modern dedektör yapısı
aruco_params = cv2.aruco.DetectorParameters()
# Yakın mesafe ve bulanıklık için eşik değerlerini optimize edelim
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10

detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# ====== 3. GÖRÜNTÜ İŞLEME DÖNGÜSÜ ======
if not os.path.exists(IMAGE_DIR):
    print(f"[HATA] '{IMAGE_DIR}' klasörü bulunamadı!")
    exit()

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"[İPUCU] Toplam {len(images)} fotoğraf bulundu. İşleniyor...")

for fname in images:
    path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(path)
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    # Modern API ile board saptama
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # En az 4 köşe bulunması kalibrasyonun sağlığı için gereklidir
    if charuco_ids is not None and len(charuco_ids) >= 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        
        # Görselleştirme
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        cv2.putText(img, f"OK: {len(charuco_ids)} corners", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon Takibi", img)
        cv2.waitKey(100)
    else:
        print(f"[UYARI] {fname}: Yetersiz köşe saptandı. Atlanıyor...")

cv2.destroyAllWindows()

# ====== 4. KALİBRASYON VE KAYIT ======
if len(all_charuco_ids) > 10 : # En az 10-15 başarılı kare önerilir
    print(f"\n[BİLGİ] {len(all_charuco_ids)} kare ile kalibrasyon başlıyor...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    if ret:
        print("\n=== KALİBRASYON BAŞARILI ===")
        print("Ortalama Hata (RMS):", ret)
        print("Kamera Matrisi:\n", camera_matrix)
        print("Bozulma Katsayıları:\n", dist_coeffs)

        np.save(os.path.join(SAVE_DIR, "calibration_matrix.npy"), camera_matrix)
        np.save(os.path.join(SAVE_DIR, "distortion_coefficients.npy"), dist_coeffs)
        print(f"\n[TAMAM] Dosyalar '{SAVE_DIR}' dizinine kaydedildi.")
    else:
        print("[HATA] Kalibrasyon hesaplanamadı.")
else:
    print(f"\n[KRİTİK HATA] Yeterli veri toplanamadı (Başarılı kare: {len(all_charuco_ids)}).")
    print("Daha net, iyi aydınlatılmış ve farklı açılardan fotoğraflar çekmeyi deneyin.")