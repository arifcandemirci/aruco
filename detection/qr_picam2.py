import cv2
from picamera2 import Picamera2

picam2 = Picamera2()

detector = cv2.QRCodeDetector()

picam2.start()

while True:

    _, img = picam2.capture_array()

    data, bbox, _ = detector.detectAndDecode(img)

    if bbox is not None:
        for i in range(len(bbox)):
            cv2.line(img, 
                     tuple(bbox[i][0]),
                     tuple(bbox[(i + 1) % len(bbox)]),
                     color = (255, 0, 0),
                     thickness = 2)
            
            cv2.putText(
                        img,
                        data,
                        (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10 ),
                        cv2.FONT_HERSHEY_COMPLEX,
                        (255, 255, 120),
                        2
                        )
            
            print ("data found: ", data)


    cv2.imshow("QR Code Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()  
    