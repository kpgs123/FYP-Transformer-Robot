import numpy as np
import cv2

# Define the size of each tile in millimeters
tile_size = 304.8

# Create arrays to store object points and image points from all the images
object_points = [] # 3D points in real world space
image_points = [] # 2D points in image plane

# Set up the camera capture object
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
cap = cv2.VideoCapture(0) # Replace with your camera source

# Loop until enough good images have been captured
num_good_images = 0
while num_good_images < 50:
    # Wait for 2 seconds
    cv2.waitKey(2000)

    # Capture a frame from the camera
    ret, img = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the tiles in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # If the corners are found, add object points and image points to the lists
    if ret == True:
        object_points.append(np.zeros((8*6, 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * tile_size
        image_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8, 6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

        num_good_images += 1

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera using the object points and image points
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:\n", camera_matrix)
np.save('camera_matrix.npy', camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)
np.save('distortion_coeffs.npy', distortion_coeffs)
