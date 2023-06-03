import cv2
from cv2 import aruco
import time
import numpy as np
import math
import serial

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

path = [[378, 249], [390, 249]]
orientations = []

# Serial communication with the robot
ser = serial.Serial('COM8', 9600, timeout=1)

# PI controller parameters
kp = 0.5
ki = 0.2
integral_sum = 0
last_error = 0

def calculate_direction(current_pos, target_pos):
    error_x = target_pos[0] - current_pos[0]
    error_y = target_pos[1] - current_pos[1]

    # PI controller
    global integral_sum, last_error
    integral_sum += (error_x + error_y)
    delta_error = (error_x + error_y) - last_error
    last_error = error_x + error_y

    control_signal = kp * (error_x + error_y) + ki * integral_sum + delta_error

    if control_signal >= 0:
        if error_x > 0:
            direction = '8'
        else:
            direction = '2'
    else:
        if error_y > 0:
            direction = '4'
        else:
            direction = '6'

    return direction

def apply_moving_average_filter(value, value_buffer):
    value_buffer.append(value)
    if len(value_buffer) > filter_size:
        value_buffer.pop(0)
    return np.mean(value_buffer, axis=0)

# Load camera calibration data
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Define the region of interest (ROI) to crop
start_x = 70  # Starting x-coordinate of the ROI
start_y = 5  # Starting y-coordinate of the ROI
end_x = 716   # Ending x-coordinate of the ROI
end_y = 589   # Ending y-coordinate of the ROI


# Moving average filter parameters
filter_size = 5
centroid_buffer = []

def apply_moving_average_filter(value, value_buffer):
    value_buffer.append(value)
    if len(value_buffer) > filter_size:
        value_buffer.pop(0)
    return np.mean(value_buffer, axis=0)

cap = cv2.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55
tolerance_margin = 5

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames

last_frame_time = time.time()

i = 0

# Define the output video file name and properties
output_file = "out1.mp4"
output_fps = 25  # Output video frames per second

# Get the first frame to extract its properties
ret, frame = cap.read()
height, width, _ = frame.shape

# Define the codec and create the video writer object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_file, fourcc, output_fps, (width, height))

while True:
    ret, frame = cap.read()

    # No more frames in video, break out of loop
    if not ret:
        break

    # Check if the frame should be processed based on the frame interval
    current_time = time.time()
    elapsed_time = current_time - last_frame_time
    if elapsed_time < frame_interval:
        time.sleep(frame_interval - elapsed_time)
        continue

    last_frame_time = current_time

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

    if len(corners) > 0:
        no_marker_count = 0
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)

        if rvec.size != 3:
            continue

        rvec = np.array(rvec).reshape((3,))
        z_rot = rvec[2]
        z_rot_deg = round(math.degrees(z_rot), 2)

        R, _ = cv2.Rodrigues(rvec)
        z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

        # Print rotation angles on the frame
        cv2.putText(cropped_frame, f"Rotation: {z_rot}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

        # Get x and y vectors in marker's coordinate system
        x_axis = np.dot(R, np.array([1, 0, 0]).T)
        y_axis = np.dot(R, np.array([0, 1, 0]).T)

        centroid = np.mean(corners[0][0], axis=0)
        centroid = apply_moving_average_filter(centroid, centroid_buffer)
        print(centroid)
        cv2.putText(cropped_frame, f"position: {centroid}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

        # Calculate direction based on error between current position and target position
        direction = calculate_direction(centroid, path[i])

        # Send direction command to the robot
        ser.write(str(direction).encode())
        while ser.readline().decode().rstrip() != 'ACK':
            pass
        
        # Check if the robot has reached the target position within the tolerance margin
        if np.linalg.norm(centroid - path[i]) < tolerance_margin:
            # Remove the reached target position from the path
            path.pop(i)
        i += 1
        frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
        cv2.imshow('frame', frame_markers)

        # Write the processed frame to the output video file
        video_writer.write(frame_markers)

    else:
        no_marker_count += 1
        if no_marker_count >= Threshold_no_marker:
            print("Cannot find ArUco marker for", Threshold_no_marker, "consecutive frames.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Send stop command to the robot
ser.write(b'0')

# Release the video capture, close the serial connection, and close all windows
cap.release()
ser.close()
cv2.destroyAllWindows()

# Release the video writer and close the output file
video_writer.release()