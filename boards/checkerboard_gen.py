import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    (5, 7),        # kare sayısı
    0.04,           # square size (m)
    0.02,           # marker size (m)
    aruco_dict
)

img = board.generateImage((2480, 3508))  # A4 @ 300 DPI
cv2.imwrite("charuco_real.png", img)
