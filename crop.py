import cv2
import numpy as np
import matplotlib.pyplot as plt

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

# Open the webcam
cap = cv2.VideoCapture(url)
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")


# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open the webcam")
    exit()

# Read a frame from the webcam
ret, frame = cap.read()

# Check if the frame is read successfully
if not ret:
    print("Failed to read the frame")
    exit()

# Undistort the frame
undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

# Convert the frame from BGR to RGB color space
undistorted_frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

# Display the undistorted frame
plt.imshow(undistorted_frame_rgb)
plt.axis('off')
plt.show()

# Get the pixel coordinates
def onclick(event):
    print("Pixel coordinates (x, y):", event.x, event.y)

# Create the interactive plot
fig, ax = plt.subplots()
ax.imshow(undistorted_frame_rgb)
ax.axis('off')
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Release the webcam
cap.release()
