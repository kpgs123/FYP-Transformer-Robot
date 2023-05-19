import cv2
import numpy as np

# Load the camera matrix and distortion coefficients
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
distortion_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Create a VideoCapture object for the default camera
# Set up the camera capture object
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
cap = cv2.VideoCapture(url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Get the camera's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the corrected video
output_path = 'corrected_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Process each frame of the video
while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    # If the frame was not read successfully, exit the loop
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coeffs)

    # Display the undistorted frame
    cv2.imshow('Undistorted Video', undistorted_frame)

    # Write the undistorted frame to the output video file
    out.write(undistorted_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
