import cv2
import numpy as np

# Load the distorted image
distorted_image = cv2.imread('G:/sem 7/FYP/New Git/FYP-Transformer-Robot/camera_image.jpg')

#distorted_image = cv2.resize(distorted_image, (100, 75))

# Define the camera matrix and distortion coefficients
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/4/camera_matrix.npy")

distortion_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/4/distortion_coeffs.npy")

# Get the image size
image_size = (distorted_image.shape[1], distorted_image.shape[0])

# Perform the undistortion
undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coeffs)

# Save the undistorted image
cv2.imwrite('undistorted_image.jpg', undistorted_image)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
