"""
Pose Estimation - tagStandard52h13 AprilTags with Raspberry Pi Camera 2

pupil-apriltags / dt-apriltags paketleri max_hamming=2 hardcode eder;
tagStandard52h13 (52-bit) için bu tablo RPi belleğini doldurur.
Bu script apriltag C kütüphanesini ctypes ile çağırır ve max_hamming=1 kullanır.

Gereksinimler:
    pip install dt-apriltags --break-system-packages   (libapriltag.so için)
"""

import ctypes
import ctypes.util
import glob
import math
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# ─── 1. libapriltag.so yükle ──────────────────────────────────────────────────
def _load_lib():
    name = ctypes.util.find_library("apriltag")
    if name:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    for pkg in ("dt_apriltags", "pupil_apriltags"):
        try:
            mod = __import__(pkg)
            d = os.path.dirname(mod.__file__)
            for path in glob.glob(os.path.join(d, "*.so*")):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
        except ImportError:
            pass
    raise RuntimeError("libapriltag bulunamadı. Kur: pip install dt-apriltags --break-system-packages")

_lib = _load_lib()
print("[INFO] Kütüphane yüklendi:", _lib._name)

# ─── 2. C struct'ları ─────────────────────────────────────────────────────────
class _ZArray(ctypes.Structure):
    _fields_ = [
        ("el_sz", ctypes.c_size_t),
        ("size",  ctypes.c_int),
        ("alloc", ctypes.c_int),
        ("data",  ctypes.c_void_p),
    ]

class _Detection(ctypes.Structure):
    # apriltag_detection_t
    _fields_ = [
        ("family",          ctypes.c_void_p),
        ("id",              ctypes.c_int),
        ("hamming",         ctypes.c_int),
        ("decision_margin", ctypes.c_float),
        ("H",               ctypes.c_void_p),
        ("c",               ctypes.c_double * 2),
        ("p",               (ctypes.c_double * 2) * 4),
    ]

class _ImageU8(ctypes.Structure):
    # image_u8_t
    _fields_ = [
        ("width",  ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("buf",    ctypes.POINTER(ctypes.c_uint8)),
    ]

# ─── 3. C fonksiyon imzaları ──────────────────────────────────────────────────
_lib.apriltag_detector_create.restype  = ctypes.c_void_p
_lib.apriltag_detector_create.argtypes = []

_lib.tagStandard52h13_create.restype  = ctypes.c_void_p
_lib.tagStandard52h13_create.argtypes = []

_lib.apriltag_detector_add_family_bits.restype  = None
_lib.apriltag_detector_add_family_bits.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
]

_lib.apriltag_detector_detect.restype  = ctypes.POINTER(_ZArray)
_lib.apriltag_detector_detect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ImageU8)]

_lib.apriltag_detections_destroy.restype  = None
_lib.apriltag_detections_destroy.argtypes = [ctypes.POINTER(_ZArray)]

# ─── 4. Detector oluştur (max_hamming=1) ──────────────────────────────────────
MAX_HAMMING = 1   # hâlâ hata verirse 0 yap

print(f"[INFO] Detector oluşturuluyor (max_hamming={MAX_HAMMING})...")
_td = _lib.apriltag_detector_create()
_tf = _lib.tagStandard52h13_create()
_lib.apriltag_detector_add_family_bits(
    ctypes.c_void_p(_td), ctypes.c_void_p(_tf), ctypes.c_int(MAX_HAMMING)
)
print("[INFO] Detector hazır")

# Detector parametrelerini doğrudan struct alanlarına yaz
# apriltag_detector_t public layout (apriltag.h): nthreads, quad_decimate,
# quad_sigma, refine_edges, decode_sharpening, debug, qtp (anonim struct), ...
class _DetPublic(ctypes.Structure):
    class _QTP(ctypes.Structure):
        _fields_ = [
            ("min_cluster_pixels",   ctypes.c_int),
            ("max_nmaxima",          ctypes.c_int),
            ("critical_rad",         ctypes.c_float),
            ("cos_critical_rad",     ctypes.c_float),
            ("max_line_fit_mse",     ctypes.c_float),
            ("min_white_black_diff", ctypes.c_int),
            ("deglitch",             ctypes.c_int),
        ]
    _fields_ = [
        ("nthreads",          ctypes.c_int),
        ("quad_decimate",     ctypes.c_float),
        ("quad_sigma",        ctypes.c_float),
        ("refine_edges",      ctypes.c_int),
        ("decode_sharpening", ctypes.c_double),
        ("debug",             ctypes.c_int),
        ("qtp",               _QTP),
    ]

_dp = ctypes.cast(_td, ctypes.POINTER(_DetPublic)).contents
_dp.nthreads          = 4
_dp.quad_decimate     = 2.0   # yarım çözünürlük → hız
_dp.quad_sigma        = 0.0
_dp.refine_edges      = 1
_dp.decode_sharpening = 0.25
_dp.debug             = 0

# ─── 5. Kalibrasyon ────────────────────────────────────────────────────────────
camera_matrix = np.load("../../calibration/calibration_matrix.npy")
dist_coeffs   = np.load("../../calibration/distortion_coefficients.npy")

TAG_SIZE = 0.020   # 2 cm

# solvePnP için tag köşe modeli (AprilTag köşe sırası: CCW, sol-alt'tan başlar)
_HALF = TAG_SIZE / 2.0
_OBJ_PTS = np.array([
    [-_HALF,  _HALF, 0],
    [ _HALF,  _HALF, 0],
    [ _HALF, -_HALF, 0],
    [-_HALF, -_HALF, 0],
], dtype=np.float64)

# ─── 6. Kamera ────────────────────────────────────────────────────────────────
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "YUV420"}
)
picam2.configure(config)
picam2.set_controls({"FrameDurationLimits": (16666, 16666)})  # 60 FPS
picam2.start()

cv2.namedWindow("AprilTag Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AprilTag Pose", 640, 480)

print("[INFO] tagStandard52h13 + pose başladı")
print("[INFO] Çıkmak için 'q'")

fps, fps_cnt, fps_t = 0.0, 0, time.perf_counter()
last_log = time.perf_counter()

try:
    while True:
        # ── Gray frame ────────────────────────────────────────────────────────
        yuv   = picam2.capture_array()
        H, W  = yuv.shape[0] * 2 // 3, yuv.shape[1]
        gray  = np.ascontiguousarray(yuv[:H, :])
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── image_u8_t (zero-copy: numpy buffer'a pointer) ────────────────────
        img = _ImageU8()
        img.width  = W
        img.height = H
        img.stride = W
        img.buf    = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # ── Detect ────────────────────────────────────────────────────────────
        dets = _lib.apriltag_detector_detect(ctypes.c_void_p(_td), ctypes.byref(img))
        n = dets.contents.size if dets else 0

        if n == 0:
            if time.perf_counter() - last_log > 1.0:
                print("[INFO] Tag bulunamadı")
                last_log = time.perf_counter()
        else:
            ptr_arr = ctypes.cast(dets.contents.data, ctypes.POINTER(ctypes.c_void_p))
            for i in range(n):
                det = ctypes.cast(ptr_arr[i], ctypes.POINTER(_Detection)).contents
                tag_id  = det.id
                corners = np.array([[det.p[j][0], det.p[j][1]] for j in range(4)], dtype=np.float32)
                center  = (int(det.c[0]), int(det.c[1]))

                # Bounding box (mavi)
                for j in range(4):
                    cv2.line(display,
                             (int(corners[j][0]),       int(corners[j][1])),
                             (int(corners[(j+1)%4][0]), int(corners[(j+1)%4][1])),
                             (255, 0, 0), 2)
                cv2.circle(display, center, 5, (0, 0, 255), -1)

                # Pose (solvePnP IPPE_SQUARE)
                ok, rvec, tvec = cv2.solvePnP(
                    _OBJ_PTS, corners,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if ok:
                    t = tvec.ravel()
                    rmat, _ = cv2.Rodrigues(rvec)
                    yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

                    # 3D eksenler
                    ax = np.float32([[0,0,0],[TAG_SIZE,0,0],[0,TAG_SIZE,0],[0,0,-TAG_SIZE]])
                    pts, _ = cv2.projectPoints(ax, rvec, tvec, camera_matrix, dist_coeffs)
                    pts = pts.astype(int)
                    o = tuple(pts[0].ravel())
                    cv2.arrowedLine(display, o, tuple(pts[1].ravel()), (0, 0, 255), 2, tipLength=0.3)
                    cv2.arrowedLine(display, o, tuple(pts[2].ravel()), (0, 255, 0), 2, tipLength=0.3)
                    cv2.arrowedLine(display, o, tuple(pts[3].ravel()), (255, 0, 0), 2, tipLength=0.3)

                    lx = int(corners[:, 0].min())
                    ly = max(int(corners[:, 1].min()) - 8, 15)
                    cv2.putText(display,
                                f"ID:{tag_id}  z={t[2]*100:.1f}cm  Yaw={yaw:.0f}",
                                (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    print(f"ID{tag_id}  x={t[0]:.3f}  y={t[1]:.3f}  z={t[2]:.3f} m  Yaw={yaw:.1f}°")

        if dets:
            _lib.apriltag_detections_destroy(dets)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_cnt += 1
        now = time.perf_counter()
        if now - fps_t >= 0.5:
            fps     = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t   = now
        cv2.putText(display, f"FPS:{fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("AprilTag Pose", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    import traceback
    print(f"[ERROR] {e}")
    traceback.print_exc()

finally:
    picam2.stop()
    cv2.destroyAllWindows()
