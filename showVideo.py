import cv2
import numpy as np
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
path = np.empty((0, 2), float)
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Define the region of interest (ROI) to crop
start_x = 100  # Starting x-coordinate of the ROI
start_y = 0  # Starting y-coordinate of the ROI
end_x = 700   # Ending x-coordinate of the ROI
end_y = 600   # Ending y-coordinate of the ROI


# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(url)  # Use 0 for the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# Read and display video frames until the user quits
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

    # If the frame was not read successfully, exit the loop
    if not ret:
        print("Failed to capture frame")
        break

    # Display the frame in a window called "Camera Feed"
    cv2.imshow("Camera Feed", cropped_frame)

    # Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
