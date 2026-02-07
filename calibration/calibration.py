import cv2
import numpy as np
import os

# ====== 1. SETTINGS AND PARAMETERS ======
BOARD_SIZE = (5, 7)
CHARUCO_SQUARE_SIZE = 0.039   # meters (39mm)
CHARUCO_MARKER_SIZE = 0.019   # meters (19mm)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250

IMAGE_DIR = "calibration_images"
SAVE_DIR = "."

# ====== 2. BOARD AND DETECTOR SETUP ======
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    CHARUCO_SQUARE_SIZE,
    CHARUCO_MARKER_SIZE,
    aruco_dict
)

# Modern detector params (tune for close-range & mild blur)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10

detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# ====== 3. IMAGE PROCESSING LOOP ======
if not os.path.exists(IMAGE_DIR):
    print(f"[ERROR] Directory not found: '{IMAGE_DIR}'")
    raise SystemExit(1)

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
images.sort()
print(f"[INFO] Found {len(images)} image(s). Processing...")

for fname in images:
    path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Could not read '{fname}'. Skipping...")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    # Detect Charuco board (modern API)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # At least 4 corners are recommended per frame for a stable calibration
    if charuco_ids is not None and len(charuco_ids) >= 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        # Visualization (optional)
        vis = img.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        cv2.putText(
            vis,
            f"OK: {len(charuco_ids)} corners",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.imshow("Calibration Tracking", vis)
        cv2.waitKey(100)
    else:
        print(f"[WARNING] {fname}: Not enough Charuco corners detected. Skipping...")

cv2.destroyAllWindows()

# ====== 4. CALIBRATION AND SAVE ======
# Recommended: at least 10-15 good frames
if len(all_charuco_ids) > 10:
    print(f"\n[INFO] Starting calibration using {len(all_charuco_ids)} valid frame(s)...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # rms is the reprojection error (lower is better)
    print("\n[SUCCESS] Calibration completed")
    print("RMS reprojection error:", rms)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    np.save(os.path.join(SAVE_DIR, "calibration_matrix.npy"), camera_matrix)
    np.save(os.path.join(SAVE_DIR, "distortion_coefficients.npy"), dist_coeffs)
    print(f"\n[SUCCESS] Saved .npy files to: '{os.path.abspath(SAVE_DIR)}'")
else:
    print(f"\n[ERROR] Not enough valid frames for calibration (valid frames: {len(all_charuco_ids)}).")
    print("[HINT] Capture more images with good lighting and different angles/distances.")
s