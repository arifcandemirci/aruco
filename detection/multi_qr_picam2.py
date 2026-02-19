import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
import time

fps = 0.0
fps_frame_count = 0
fps_time = time.perf_counter()
last_log = time.perf_counter()

detector = cv2.QRCodeDetector()

#Starting Camera
picam2 = Picamera2()


config =picam2.create_preview_configuration(
    transform=Transform(hflip=True, vflip=True),
    main={"size": (320, 240), "format": "RGB888"},  #for imshow
    lores={"size": (320, 240), "format": "YUV420"}  #for analysis
    )

picam2.configure(config)

picam2.set_controls({"FrameDurationLimits": (8888, 8888)}) #30 Fps
picam2.start()

try:
    while True:
        frame_main = picam2.capture_array("main")
        frame_lores = picam2.capture_array("lores")

        h = frame_lores.shape[0] * 2 // 3   # 2/3 of the data is grey data in YUV420  
        grey = frame_lores[0:h, :]        #so 320 * 2 / 3 is equal to 240
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(grey)

        if retval and points is not None:
            
            scale_x = frame_main.shape[1] / grey.shape[1]   #scaling the diffrence
            scale_y = frame_main.shape[0] / grey.shape[0]   #between main and lores

            for i in range(len(decoded_info)):

                data = decoded_info[i]

                pts = points[i].copy()          
                pts[:, 0] *= scale_x      
                pts[:, 1] *= scale_y 
                pts = pts.astype(int)

                for j in range(4):
                     
                     cv2.line(frame_main,
                              tuple(pts[j]),
                              tuple(pts[(j+1) % 4]),
                              (255, 0, 0), 2)
                     
                cv2.putText(frame_main,
                            data,
                            tuple(pts[0]),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 255, 120),
                            2)
                
                print ("Data Found: ", data)
        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.5:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_time = now
        cv2.putText(frame_main, f"FPS:{fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        cv2.imshow("QR Code Detector", frame_main) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
     print(f"[ERROR] {e}")

finally:  

    picam2.stop()
    cv2.destroyAllWindows()