import cv2
import numpy as np
# Open the video capture for webcam

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
video = cv2.VideoCapture(url)


# Load camera matrix and distortion matrix from numpy arrays

camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
distortion_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")



# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the region of interest (ROI) to crop
start_x = 75  # Starting x-coordinate of the ROI
start_y = 5  # Starting y-coordinate of the ROI
end_x = 706   # Ending x-coordinate of the ROI
end_y = 589   # Ending y-coordinate of the ROI

# Create a VideoWriter object to save the cropped video
output = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (end_x - start_x, end_y - start_y))

while True:
    # Read a frame from the video capture
    ret, frame = video.read()
    if not ret:
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_matrix)

    # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

    # Write the cropped frame to the output video
    output.write(cropped_frame)

    # Display the cropped frame
    cv2.imshow('Cropped Video', cropped_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects, and close windows
video.release()
output.release()
cv2.destroyAllWindows()
