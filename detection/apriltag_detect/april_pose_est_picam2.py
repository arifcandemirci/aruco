"""
Pose Estimation using tagStandard52h13 AprilTag markers with Raspberry Pi Camera 2

Requires:
    pip install pupil-apriltags
    (or: pip install dt-apriltags)
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import math
from dt_apriltags import Detector

# ─── Calibration ─────────────────────────────────────────────────────────────
camera_matrix = np.load("../../calibration/calibration_matrix.npy")
dist_coeffs   = np.load("../../calibration/distortion_coefficients.npy")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# ─── AprilTag settings ────────────────────────────────────────────────────────
TAG_FAMILY     = "tagStandard52h13"
TAG_SIZE       = 0.020          # marker side length in metres (2 cm)

detector = Detector(
    families=TAG_FAMILY,
    nthreads=2,
    quad_decimate=2.0,    # 2.0 = yarım çözünürlükte işle → daha hızlı, RPi dostu
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
)

# ─── Camera ───────────────────────────────────────────────────────────────────
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "YUV420"}
)
picam2.configure(config)
picam2.set_controls({"FrameDurationLimits": (16666, 16666)})  # 60 FPS
picam2.start()

cv2.namedWindow("AprilTag Pose (Camera)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AprilTag Pose (Camera)", 640, 480)

print("[INFO] tagStandard52h13 detection + pose started")
print(f"[INFO] Tag family : {TAG_FAMILY}")
print(f"[INFO] Tag size   : {TAG_SIZE * 100:.1f} cm")
print("[INFO] Press 'q' to quit")

fps            = 0.0
fps_frame_cnt  = 0
fps_time       = time.perf_counter()
last_log       = time.perf_counter()

try:
    while True:
        # ── Grab greyscale frame from YUV420 ──────────────────────────────────
        frame_yuv = picam2.capture_array()
        h_yuv     = frame_yuv.shape[0]
        h_gray    = h_yuv * 2 // 3          # Y plane occupies top 2/3 of YUV420
        gray      = frame_yuv[:h_gray, :]   # shape: (480, 640), dtype: uint8

        # For display we use the raw gray image and convert to BGR
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── Detect AprilTags ──────────────────────────────────────────────────
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=TAG_SIZE,
        )

        if detections:
            for det in detections:
                tag_id = det.tag_id

                # ── Draw bounding box (blue) ──────────────────────────────────
                corners = det.corners.astype(int)           # (4, 2)
                for j in range(4):
                    cv2.line(display,
                             tuple(corners[j]),
                             tuple(corners[(j + 1) % 4]),
                             (255, 0, 0), 2)

                # ── Draw centre dot ───────────────────────────────────────────
                cx_tag = int(det.center[0])
                cy_tag = int(det.center[1])
                cv2.circle(display, (cx_tag, cy_tag), 4, (0, 0, 255), -1)

                # ── Pose from detector ────────────────────────────────────────
                if det.pose_t is not None and det.pose_R is not None:
                    t    = det.pose_t.ravel()          # translation  [x, y, z]
                    rmat = det.pose_R                  # 3×3 rotation matrix

                    # ── Draw 3-D axes (project manually) ──────────────────────
                    rvec, _ = cv2.Rodrigues(rmat)
                    axis_len = TAG_SIZE
                    axis_pts = np.float32([
                        [0,        0,        0       ],
                        [axis_len, 0,        0       ],
                        [0,        axis_len, 0       ],
                        [0,        0,       -axis_len],
                    ])
                    img_pts, _ = cv2.projectPoints(
                        axis_pts, rvec, det.pose_t,
                        camera_matrix, dist_coeffs
                    )
                    img_pts = img_pts.astype(int)

                    origin  = tuple(img_pts[0].ravel())
                    x_tip   = tuple(img_pts[1].ravel())
                    y_tip   = tuple(img_pts[2].ravel())
                    z_tip   = tuple(img_pts[3].ravel())

                    cv2.arrowedLine(display, origin, x_tip, (0,   0,   255), 2, tipLength=0.3)  # X red
                    cv2.arrowedLine(display, origin, y_tip, (0,   255, 0  ), 2, tipLength=0.3)  # Y green
                    cv2.arrowedLine(display, origin, z_tip, (255, 0,   0  ), 2, tipLength=0.3)  # Z blue

                    # ── Yaw from rotation matrix ─────────────────────────────
                    yaw_deg = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

                    info = (f"ID{tag_id} x={t[0]:.3f} y={t[1]:.3f} "
                            f"z={t[2]:.3f} m  Yaw={yaw_deg:.1f}deg")
                    print(info)

                    # ── Overlay text on frame ──────────────────────────────────
                    label_y = corners[:, 1].min() - 8
                    cv2.putText(display,
                                f"ID:{tag_id}  z={t[2]*100:.1f}cm  Yaw={yaw_deg:.0f}",
                                (corners[:, 0].min(), max(label_y, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Not detected — print a gentle reminder every second
            if time.perf_counter() - last_log > 1.0:
                print("[INFO] No AprilTag detected")
                last_log = time.perf_counter()

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_frame_cnt += 1
        now     = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.5:
            fps           = fps_frame_cnt / elapsed
            fps_frame_cnt = 0
            fps_time      = now
        cv2.putText(display, f"FPS:{fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("AprilTag Pose (Camera)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
