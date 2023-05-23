import cv2
import numpy as np
import matplotlib.pyplot as plt

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

# Load camera matrix and distortion matrix from numpy arrays
camera_matrix = np.load('camera.npy')
distortion_matrix = np.load('distortion.npy')

# Open the video capture for webcam
video = cv2.VideoCapture(url)
# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the region of interest (ROI) to crop
start_x = 100  # Starting x-coordinate of the ROI
start_y = 100  # Starting y-coordinate of the ROI
end_x = 400    # Ending x-coordinate of the ROI
end_y = 300    # Ending y-coordinate of the ROI


while True:
    # Read a frame from the video capture
    ret, frame = video.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_matrix)

    # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]



    # Display the cropped frame
    cv2.imshow('Cropped Video', cropped_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and close windows
video.release()
output.release()
cv2.destroyAllWindows()
