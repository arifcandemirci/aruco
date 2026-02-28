import cv2

cap = cv2.VideoCapture(0) #Might be 0, 1, 2

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

detector = cv2.QRCodeDetector()

if not cap.isOpened():
    print("[ERROR] Camera could not be opened")
    exit()

print("[INFO] Starting live QR Code detection...")
print("[INFO] Press 'q' to quit")

while True:

    ret, img = cap.read()

    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)

    if retval is True:
            
            for i in range(len(decoded_info)):

                data = decoded_info[i]
                pts = points[i].astype(int)

                for j in range(4):
                     
                     cv2.line(img,
                              tuple(pts[j]),
                              tuple(pts[(j+1) % 4]),
                              (255, 0, 0), 2)
                     
                cv2.putText(img,
                            data,
                            tuple(pts[0]),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 255, 120),
                            2)
                
                print ("Data Found: ", data)

    cv2.imshow("QR Code Detector", img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()       
cv2.destroyAllWindows()

