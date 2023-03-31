import cv2
import cv2.aruco as aruco
import numpy as np

# Load camera calibration parameters
camera_matrix = np.load("calibration_matrix.npy")
dist_coeffs = np.load("distortion_coefficients.npy")

# Define dictionary of ArUco markers
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

# Define parameters for detection
parameters = aruco.DetectorParameters_create()

# Load image from camera or file
image = cv2.imread("image.jpg")

# Detect markers in image
corners, ids, rejected_img_points = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

# If markers are detected, estimate pose
if ids is not None:
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
    for i in range(ids.size):
        # Draw axis and bounding box on image
        aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
        aruco.drawDetectedMarkers(image, corners)
    
# Display image with pose estimation
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
