import cv2
import numpy as np

# Define marker parameters
marker_size = 200  # in pixels
marker_id = 23

# Create dictionary of ArUco markers
#aruco_dict = cv2.aruco.Dictionary_create(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.Dictionary_create(cv2.aruco.DICT_6X6_250, marker_size)


# Generate marker image
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_image, 1)

# Save marker image to file
cv2.imwrite("marker{}.png".format(marker_id), marker_image)
