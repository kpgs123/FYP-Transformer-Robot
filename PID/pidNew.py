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
#path = [(350, 150), (360, 150), (370, 150), (380, 150), (390, 150), (400, 150)]

path = [[228, 170], [230, 170], [232, 170], [234, 170], [236, 170], [238, 170], [240, 170], [242, 170], [244, 170], [246, 170], [248, 170], [250, 170], [252, 170], [254, 170], [256, 170], [258, 170], [260, 170], [262, 170], [264, 170], [266, 170], [268, 170], [270, 170], [272, 170], [274, 170], [276, 170], [278, 170], [280, 170], [282, 170], [284, 170], [286, 170], [288, 170], [290, 170], [292, 170], [294, 170], [296, 170], [298, 170], [300, 170], [302, 170], [304, 170], [306, 170], [308, 170], [310, 170], [312, 170], [314, 170], [316, 170], [318, 170], [320, 170], [322, 170], [324, 170], [326, 
170]]

path = [[378,268], [390,268]]
orientations = []

# Serial communication with the robot
ser = serial.Serial('COM8', 9600, timeout=1)

# PI controller parameters
kp = 0.5
ki = 0.2
integral_sum = 0
last_error = 0

# Calculate direction based on error between current position and target position
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

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames

last_frame_time = time.time()

while True:
    ret, frame = cap.read()

    # No more frames in video, break out of loop
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - last_frame_time
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

    # Check if the frame should be processed based on the frame interval
    if current_time - last_frame_time >= frame_interval:
        last_frame_time = current_time

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
            cv2.putText(cropped_frame, f"position: {centroid}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

            # Calculate direction based on error between current position and target position
            direction = calculate_direction(centroid, path[0])

    

            # Send direction command to the robot
            ser.write(str(direction).encode())
            while ser.readline().decode().rstrip() != 'ACK':
              pass
            

            # Check if the robot has reached the target position
            if centroid[0] == path[0][0] and centroid[1] == path[0][1]:
                path.pop(0)  # Remove the reached target position from the path

            frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
            cv2.imshow('frame', frame_markers)

        else:
            no_marker_count += 1
            if no_marker_count >= Threshold_no_marker:
                print("Cannot find ArUco marker for", Threshold_no_marker, "consecutive frames.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

cap.release()
ser.close()
print(path)
