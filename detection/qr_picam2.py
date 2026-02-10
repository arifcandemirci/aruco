import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()

detector = cv2.QRCodeDetector()

fps = 0.0
fps_frame_count = 0
fps_time = time.perf_counter()
last_log = time.perf_counter()

picam2.start()

while True:

    img = picam2.capture_array()

    data, bbox, _ = detector.detectAndDecode(img)

    if bbox is not None:
        for i in range(len(bbox[0])):
            cv2.line(img, 
                     (int(bbox[0][i][0]), int(bbox[0][i][1])),  #x1,y1
                     (int(bbox[0][(i+1) % 4][0]), int(bbox[0][(i+1) % 4][1])),  #x2,y2                   
                     color = (255, 0, 0),
                     thickness = 2)
            
            cv2.putText(
                        img,
                        data,
                        (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10 ),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0,
                        (255, 255, 120),
                        2
                        )
            
            print ("data found: ", data)

    fps_frame_count += 1
    now = time.perf_counter()
    elapsed = now - fps_time
    if elapsed >= 0.5:
        fps = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_time = now
    cv2.putText(output, f"FPS:{fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("QR Code Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()