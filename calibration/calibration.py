from pathlib import Path

import cv2
import numpy as np

# 1. SETTINGS AND PARAMETERS
BOARD_SIZE = (5, 7)
CHARUCO_SQUARE_SIZE = 0.039  # meters (39 mm)
CHARUCO_MARKER_SIZE = 0.019  # meters (19 mm)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_250

CALIBRATION_DIR = Path(__file__).resolve().parent
IMAGE_DIR = CALIBRATION_DIR / "calibration_images"
SAVE_DIR = CALIBRATION_DIR

# 2. BOARD AND DETECTOR SETUP
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    CHARUCO_SQUARE_SIZE,
    CHARUCO_MARKER_SIZE,
    aruco_dict,
)

aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10

detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# 3. IMAGE PROCESSING LOOP
if not IMAGE_DIR.exists():
    print(f"[ERROR] Directory not found: '{IMAGE_DIR}'")
    raise SystemExit(1)

images = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
)
print(f"[INFO] Found {len(images)} image(s). Processing...")

for path in images:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARNING] Could not read '{path.name}'. Skipping...")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) >= 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        vis = img.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        cv2.putText(
            vis,
            f"OK: {len(charuco_ids)} corners",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration Tracking", vis)
        cv2.waitKey(100)
    else:
        print(
            f"[WARNING] {path.name}: Not enough ChArUco corners detected. "
            "Skipping..."
        )

cv2.destroyAllWindows()

# 4. CALIBRATION AND SAVE
if len(all_charuco_ids) > 10:
    print(
        f"\n[INFO] Starting calibration using "
        f"{len(all_charuco_ids)} valid frame(s)..."
    )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    print("\n[SUCCESS] Calibration completed")
    print("RMS reprojection error:", rms)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    np.save(SAVE_DIR / "calibration_matrix.npy", camera_matrix)
    np.save(SAVE_DIR / "distortion_coefficients.npy", dist_coeffs)
    print(f"\n[SUCCESS] Saved calibration files to: '{SAVE_DIR}'")
else:
    print(
        f"\n[ERROR] Not enough valid frames for calibration "
        f"(valid frames: {len(all_charuco_ids)})."
    )
    print("[HINT] Capture more images with good lighting and varied angles/distances.")
