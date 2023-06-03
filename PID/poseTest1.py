import cv2
from cv2 import aruco
import time
import numpy as np
import math

# ArUco dictionary and parameters
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Camera calibration matrices
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Region of interest (ROI) coordinates
start_x = 70
start_y = 5
end_x = 716
end_y = 589

# Moving average filter parameters
filter_size = 5
centroid_buffer = []

# Video capture
cap = cv2.VideoCapture("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi")

# Frame processing parameters
fps_limit = 10
frame_interval = 1 / fps_limit
last_frame_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    if current_time - last_frame_time >= frame_interval:
        last_frame_time = current_time

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
            rvec = np.array(rvec).reshape((3,))
            R, _ = cv2.Rodrigues(rvec)
            z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

            centroid = np.mean(corners[0][0], axis=0)
            centroid_buffer.append(centroid)
            if len(centroid_buffer) > filter_size:
                centroid_buffer.pop(0)

            frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
            cv2.putText(frame_markers, f"Rotation: {z_rot}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 200), 2)
            cv2.putText(frame_markers, f"Position: {centroid_buffer[-1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 25, 59), 2)

            cv2.imshow('frame', frame_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

cap.release()
