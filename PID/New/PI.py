import cv2
from cv2 import aruco
import time
import numpy as np
import math
import serial

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

# Robot control parameters
k_p_horizontal = 0.2
k_i_horizontal = 0.01
error_sum_horizontal = 0

k_p_vertical = 0.2
k_i_vertical = 0.01
error_sum_vertical = 0

k_p_diagonal = 0.2
k_i_diagonal = 0.01
error_sum_diagonal = 0

# Serial communication with the robot
ser = serial.Serial('COM8', 9600, timeout=1)

def send_command_to_esp32(command):
    ser.write(command.encode())
    response = ser.readline().decode().rstrip()
    if response:
        print(response)

# Function to calculate movement commands based on coordinates
def move_robot_to_coordinates(x, y):
    global error_sum_horizontal, error_sum_vertical, error_sum_diagonal

    # Calculate the necessary movement commands to reach the given coordinates

    # Move diagonally
    if x < 378 and y < 249:
        command = '7'  # Diagonal top-left
        error_x = x - 378
        error_y = y - 249
        error_sum_diagonal += error_x + error_y
        command += str(round(k_p_diagonal * (error_x + error_y) + k_i_diagonal * error_sum_diagonal))
    elif x > 430 and y < 249:
        command = '9'  # Diagonal top-right
        error_x = x - 430
        error_y = y - 249
        error_sum_diagonal += error_x + error_y
        command += str(round(k_p_diagonal * (error_x + error_y) + k_i_diagonal * error_sum_diagonal))
    elif x < 378 and y > 249:
        command = '1'  # Diagonal bottom-left
        error_x = x - 378
        error_y = y - 249
        error_sum_diagonal += error_x + error_y
        command += str(round(k_p_diagonal * (error_x + error_y) + k_i_diagonal * error_sum_diagonal))
    elif x > 430 and y > 249:
        command = '3'  # Diagonal bottom-right
        error_x = x - 430
        error_y = y - 249
        error_sum_diagonal += error_x + error_y
        command += str(round(k_p_diagonal * (error_x + error_y) + k_i_diagonal * error_sum_diagonal))
    # Move horizontally
    elif x < 378:
        command = '4'  # Left
        error_x = x - 378
        error_sum_horizontal += error_x
        command += str(round(k_p_horizontal * error_x + k_i_horizontal * error_sum_horizontal))
    elif x > 430:
        command = '6'  # Right
        error_x = x - 430
        error_sum_horizontal += error_x
        command += str(round(k_p_horizontal * error_x + k_i_horizontal * error_sum_horizontal))
    # Move vertically
    else:
        command = '8'  # Forward
        error_y = y - 249
        error_sum_vertical += error_y
        command += str(round(k_p_vertical * error_y + k_i_vertical * error_sum_vertical))

    send_command_to_esp32(command)  # Send the movement command

# Video capture
cap = cv2.VideoCapture("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi")

# Frame processing parameters
fps_limit = 10
frame_interval = 1 / fps_limit
last_frame_time = time.time()

# Path to follow
path = [[378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249], [378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249]]

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

            # Check if the robot has reached the target position with a tolerance
            if len(path) > 0:
                target_x, target_y = path[0]
                current_x, current_y = centroid_buffer[-1]
                if abs(target_x - current_x) <= 5 and abs(target_y - current_y) <= 5:
                    target_reached = True
                    path.pop(0)  # Remove the reached target from the path

                    # Reset the error sums for smooth movement to the next target
                    error_sum_horizontal = 0
                    error_sum_vertical = 0
                    error_sum_diagonal = 0
                else:
                    target_reached = False

            # Move the robot to the next target position
            if not target_reached:
                move_robot_to_coordinates(target_x, target_y)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

cap.release()
