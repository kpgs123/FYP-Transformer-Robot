import serial
import time
import cv2
from cv2 import aruco
import numpy as np
import math

# Serial communication with the robot
ser = serial.Serial('COM8', 9600, timeout=1)

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

# Path coordinates
path = [[378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249], [378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249]]

# Tolerance for reaching a position
tolerance = 10  # Adjust the tolerance as per your requirements

# Frame processing parameters
fps_limit = 10
frame_interval = 1 / fps_limit
last_frame_time = time.time()

# Function to send movement command to the robot
def send_command_to_esp32(command):
    ser.write(command.encode())
    response = ser.readline().decode().rstrip()
    if response:
        print(response)

# Function to calculate movement commands based on coordinates
def move_robot_to_coordinates(x, y):
    # Calculate the necessary movement commands to reach the given coordinates

    # Move diagonally
    if x < 378 and y < 249:
        command = '7'  # Diagonal top-left
    elif x > 430 and y < 249:
        command = '9'  # Diagonal top-right
    elif x < 378 and y > 249:
        command = '1'  # Diagonal bottom-left
    elif x > 430 and y > 249:
        command = '3'  # Diagonal bottom-right
    # Move horizontally
    elif x < 378:
        command = '4'  # Left
    elif x > 430:
        command = '6'  # Right
    else:
        command = ''  # Stay in the same position horizontally

    send_command_to_esp32(command)  # Send the horizontal/diagonal movement command

    # Move vertically
    if y < 249:
        command = '2'  # Backward
    elif y > 249:
        command = '8'  # Forward
    else:
        command = ''  # Stay in the same position vertically

    send_command_to_esp32(command)  # Send the vertical movement command

# Video capture
cap = cv2.VideoCapture("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi")

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

            if len(path) > 0:
                x, y = path[0]  # Get the next target coordinates from the path
                target_reached = False

                while not target_reached:
                    move_robot_to_coordinates(x, y)

                    # Check if the robot has reached the target position within the tolerance
                    current_x, current_y = centroid_buffer[-1]
                    if abs(current_x - x) <= tolerance and abs(current_y - y) <= tolerance:
                        target_reached = True

                path.pop(0)  # Remove the visited target from the path

                while ser.readline().decode().rstrip() != 'ACK':
                    pass
    if cv2.waitKey(1) & 0xFF == ord('q') or len(path) == 0:
        break

if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

cap.release()
