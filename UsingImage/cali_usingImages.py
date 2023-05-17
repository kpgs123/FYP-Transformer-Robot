import glob
import numpy as np
import cv2

# Define the number of corners in the checkerboard
num_corners_x = 5
num_corners_y = 3

# Define the size of each square on the checkerboard in millimeters
square_size = 45.0

# Create arrays to store object points and image points from all the images
object_points = []  # 3D points in real-world space
image_points = []  # 2D points in image plane

# Get the filenames of the saved calibration images
calibration_images = glob.glob('G:/sem 7/FYP/New Git/FYP-Transformer-Robot/UsingImage/cali_*.jpg')
i=0
# Loop through the calibration images
for image_file in calibration_images:
    # Read the image
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

    # If the corners are found, add object points and image points to the lists
    if ret == True:
        object_points.append(np.zeros((num_corners_x * num_corners_y, 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2) * square_size
        image_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (num_corners_x, num_corners_y), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        i+=1
        print(i)

# Close the OpenCV windows
cv2.destroyAllWindows()

# Calibrate the camera using the object points and image points
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:\n", camera_matrix)
np.save('camera_matrix.npy', camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)
np.save('distortion_coeffs.npy', distortion_coeffs)
