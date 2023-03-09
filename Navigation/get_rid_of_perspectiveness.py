import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("chess4.png")
plt.imshow(img)
plt.show()

pattern_size = (7, 7)
'''
for i in range(5, 15):
    print(i)
    for j in range(5, 15):
        pattern_size = (i, j)
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            print("%d, %d" % pattern_size)
'''

found, corners = cv.findChessboardCorners(img, pattern_size)

if found:
    h, w = img.shape[:2]
    src_points = np.float32([corners[0], corners[pattern_size[0] - 1], corners[-1], corners[-pattern_size[0]]])
    dst_points = np.float32([[0, 0], [w,0], [w,h], [0,h]])
    M = cv.getPerspectiveTransform(src_points, dst_points)
    img_warped = cv.warpPerspective(img, M, (w,h))
    cv.imshow("Transforme Image", img_warped)
    cv.waitKey(0)
else:
    print("Chessboard pattern not found")