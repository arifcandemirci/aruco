"""
Pose Estimation - tagStandard52h13 AprilTags with Raspberry Pi Camera 2

The detector directly interfaces with libapriltag through ctypes so the
maximum Hamming distance can be constrained for embedded-memory use.
"""

from pathlib import Path
import ctypes
import ctypes.util
import glob
import math
import os
import time

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2


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
            package_dir = os.path.dirname(mod.__file__)
            for path in glob.glob(os.path.join(package_dir, "*.so*")):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
        except ImportError:
            pass

    raise RuntimeError(
        "libapriltag was not found. Install an AprilTag implementation "
        "such as dt-apriltags or provide libapriltag.so on the system."
    )


_lib = _load_lib()
print("[INFO] Loaded AprilTag library:", _lib._name)


class _ZArray(ctypes.Structure):
    _fields_ = [
        ("el_sz", ctypes.c_size_t),
        ("size", ctypes.c_int),
        ("alloc", ctypes.c_int),
        ("data", ctypes.c_void_p),
    ]


class _Detection(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_void_p),
        ("id", ctypes.c_int),
        ("hamming", ctypes.c_int),
        ("decision_margin", ctypes.c_float),
        ("H", ctypes.c_void_p),
        ("c", ctypes.c_double * 2),
        ("p", (ctypes.c_double * 2) * 4),
    ]


class _ImageU8(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
    ]


_lib.apriltag_detector_create.restype = ctypes.c_void_p
_lib.apriltag_detector_create.argtypes = []
_lib.tagStandard52h13_create.restype = ctypes.c_void_p
_lib.tagStandard52h13_create.argtypes = []
_lib.apriltag_detector_add_family_bits.restype = None
_lib.apriltag_detector_add_family_bits.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
_lib.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
_lib.apriltag_detector_detect.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_ImageU8),
]
_lib.apriltag_detections_destroy.restype = None
_lib.apriltag_detections_destroy.argtypes = [ctypes.POINTER(_ZArray)]

MAX_HAMMING = 1
print(f"[INFO] Creating detector (max_hamming={MAX_HAMMING})...")
_td = _lib.apriltag_detector_create()
_tf = _lib.tagStandard52h13_create()
_lib.apriltag_detector_add_family_bits(
    ctypes.c_void_p(_td),
    ctypes.c_void_p(_tf),
    ctypes.c_int(MAX_HAMMING),
)


class _DetPublic(ctypes.Structure):
    class _QTP(ctypes.Structure):
        _fields_ = [
            ("min_cluster_pixels", ctypes.c_int),
            ("max_nmaxima", ctypes.c_int),
            ("critical_rad", ctypes.c_float),
            ("cos_critical_rad", ctypes.c_float),
            ("max_line_fit_mse", ctypes.c_float),
            ("min_white_black_diff", ctypes.c_int),
            ("deglitch", ctypes.c_int),
        ]

    _fields_ = [
        ("nthreads", ctypes.c_int),
        ("quad_decimate", ctypes.c_float),
        ("quad_sigma", ctypes.c_float),
        ("refine_edges", ctypes.c_int),
        ("decode_sharpening", ctypes.c_double),
        ("debug", ctypes.c_int),
        ("qtp", _QTP),
    ]


_dp = ctypes.cast(_td, ctypes.POINTER(_DetPublic)).contents
_dp.nthreads = 4
_dp.quad_decimate = 2.0
_dp.quad_sigma = 0.0
_dp.refine_edges = 1
_dp.decode_sharpening = 0.25
_dp.debug = 0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
camera_matrix = np.load(CALIBRATION_DIR / "calibration_matrix.npy")
dist_coeffs = np.load(CALIBRATION_DIR / "distortion_coefficients.npy")

TAG_SIZE = 0.020
_HALF = TAG_SIZE / 2.0
_OBJ_PTS = np.array(
    [
        [-_HALF, _HALF, 0],
        [_HALF, _HALF, 0],
        [_HALF, -_HALF, 0],
        [-_HALF, -_HALF, 0],
    ],
    dtype=np.float64,
)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    transform=Transform(hflip=True, vflip=True),
    main={"size": (640, 480), "format": "YUV420"},
)
picam2.configure(config)
picam2.set_controls({"FrameDurationLimits": (16666, 16666)})
picam2.start()

cv2.namedWindow("AprilTag Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AprilTag Pose", 640, 480)

print("[INFO] tagStandard52h13 pose estimation started")
print("[INFO] Press 'q' to quit")

fps = 0.0
fps_count = 0
fps_time = time.perf_counter()
last_log = time.perf_counter()

try:
    while True:
        yuv = picam2.capture_array()
        height = yuv.shape[0] * 2 // 3
        width = yuv.shape[1]
        gray = np.ascontiguousarray(yuv[:height, :])
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        image = _ImageU8()
        image.width = width
        image.height = height
        image.stride = width
        image.buf = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        detections = _lib.apriltag_detector_detect(
            ctypes.c_void_p(_td), ctypes.byref(image)
        )
        count = detections.contents.size if detections else 0

        if count == 0:
            if time.perf_counter() - last_log > 1.0:
                print("[INFO] No tag detected")
                last_log = time.perf_counter()
        else:
            ptr_array = ctypes.cast(
                detections.contents.data,
                ctypes.POINTER(ctypes.c_void_p),
            )

            for i in range(count):
                detection = ctypes.cast(
                    ptr_array[i], ctypes.POINTER(_Detection)
                ).contents
                tag_id = detection.id
                corners = np.array(
                    [
                        [detection.p[j][0], detection.p[j][1]]
                        for j in range(4)
                    ],
                    dtype=np.float32,
                )
                center = (int(detection.c[0]), int(detection.c[1]))

                for j in range(4):
                    cv2.line(
                        display,
                        (int(corners[j][0]), int(corners[j][1])),
                        (
                            int(corners[(j + 1) % 4][0]),
                            int(corners[(j + 1) % 4][1]),
                        ),
                        (255, 0, 0),
                        2,
                    )
                cv2.circle(display, center, 5, (0, 0, 255), -1)

                ok, rvec, tvec = cv2.solvePnP(
                    _OBJ_PTS,
                    corners,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if ok:
                    translation = tvec.ravel()
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    yaw = math.degrees(
                        math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    )

                    axes = np.float32(
                        [
                            [0, 0, 0],
                            [TAG_SIZE, 0, 0],
                            [0, TAG_SIZE, 0],
                            [0, 0, -TAG_SIZE],
                        ]
                    )
                    points, _ = cv2.projectPoints(
                        axes,
                        rvec,
                        tvec,
                        camera_matrix,
                        dist_coeffs,
                    )
                    points = points.astype(int)
                    origin = tuple(points[0].ravel())
                    cv2.arrowedLine(
                        display,
                        origin,
                        tuple(points[1].ravel()),
                        (0, 0, 255),
                        2,
                        tipLength=0.3,
                    )
                    cv2.arrowedLine(
                        display,
                        origin,
                        tuple(points[2].ravel()),
                        (0, 255, 0),
                        2,
                        tipLength=0.3,
                    )
                    cv2.arrowedLine(
                        display,
                        origin,
                        tuple(points[3].ravel()),
                        (255, 0, 0),
                        2,
                        tipLength=0.3,
                    )

                    label_x = int(corners[:, 0].min())
                    label_y = max(int(corners[:, 1].min()) - 8, 15)
                    cv2.putText(
                        display,
                        f"ID:{tag_id} z={translation[2] * 100:.1f}cm Yaw={yaw:.0f}",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    print(
                        f"ID {tag_id} | x={translation[0]:.3f} m "
                        f"y={translation[1]:.3f} m z={translation[2]:.3f} m "
                        f"Yaw={yaw:.1f} deg"
                    )

        if detections:
            _lib.apriltag_detections_destroy(detections)

        fps_count += 1
        now = time.perf_counter()
        if now - fps_time >= 0.5:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        cv2.putText(
            display,
            f"FPS:{fps:.1f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow("AprilTag Pose", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as exc:
    import traceback

    print(f"[ERROR] {exc}")
    traceback.print_exc()

finally:
    picam2.stop()
    cv2.destroyAllWindows()
