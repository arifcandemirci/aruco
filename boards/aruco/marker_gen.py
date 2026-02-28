import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker_id = 0      # marker numarası
size_px  = 800     # piksel boyutu

img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
cv2.imwrite("markers/marker_0.png", img)
