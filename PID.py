import numpy as np
import cv2 as cv
import time
import math
from cv2 import aruco


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        #self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0

    def calculate(self, error, dt):
        # Proportional term
        proportional = self.Kp * error

        # Integral term
        self.integral += error * dt
        #integral = self.Ki * self.integral

        # Derivative term
        derivative = self.Kd * (error - self.last_error) / dt
        self.last_error = error

        # Calculate control output
        #control_output = proportional + integral + derivative
        control_output = proportional + derivative

        return control_output

# Example usage
# Assuming you have the desired_pixel_coords and current_pixel_coords


dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
#url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
path = np.empty((0, 2), float)
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Define the region of interest (ROI) to crop
start_x = 100  # Starting x-coordinate of the ROI
start_y = 0  # Starting y-coordinate of the ROI
end_x = 700   # Ending x-coordinate of the ROI
end_y = 600   # Ending y-coordinate of the ROI



cap = cv.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames


dt = 0.001

# PID gains
Kp = 0.5
#Ki = 0.1
Kd = 0.2

# Create PID controller
pid_controller = PIDController(Kp, Ki, Kd)

# Desired pixel coordinates
desired_pixel_coords = (300, 300)

# Loop (e.g., in a robot control loop)
last_frame_time = time.time()
while True:
    # Get current pixel coordinates from the camera
    ret, frame = cap.read()
    # No more frames in video, break out of loop
    if not ret:
        break

    current_time = time.time()
    # Undistort the frame
    undistorted_frame = cv.undistort(frame, camera_matrix, dist_coeffs)

    # Crop the undistorted frame
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]
     # Check if the frame should be processed based on the frame interval
    if current_time - last_frame_time >= frame_interval:
        last_frame_time = current_time

        gray = cv.cvtColor(cropped_frame, cv.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            no_marker_count = 0
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)

            if rvec.size != 3:
                continue

            rvec = np.array(rvec).reshape((3,))
            z_rot = rvec[2]
            z_rot_deg = round(math.degrees(z_rot), 2)

            R, _ = cv.Rodrigues(rvec)
            z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

            # Print rotation angles on the frame
            cv.putText(cropped_frame, f"Rotation: {z_rot}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

            # Get x and y vectors in marker's coordinate system
            x_axis = np.dot(R, np.array([1, 0, 0]).T)
            y_axis = np.dot(R, np.array([0, 1, 0]).T)

            centroid = np.mean(corners[0][0], axis=0)
            current_pixel_coords = centroid[::-1]

            # Calculate error in pixel coordinates
            error_x = desired_pixel_coords[0] - current_pixel_coords[0]
            error_y = desired_pixel_coords[1] - current_pixel_coords[1]

            # Calculate control output using PID controller
            control_output_x = pid_controller.calculate(error_x, dt)
            control_output_y = pid_controller.calculate(error_y, dt)

            print(control_output_x, control_output_y)
            

            # Apply the control outputs to control the robot's position or movement
            #move_robot(control_output_x, control_output_y)

            # Sleep or delay for a specific time (dt) before the next iteration
            time.sleep(dt)


            cv.putText(cropped_frame, f"position: {centroid}", (10,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

            path = np.append(path , np.array([centroid]), axis=0)

            frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
            cv.imshow('frame', frame_markers)

        else:
            no_marker_count += 1
            if no_marker_count >= Threshold_no_marker:
                print("Cannot find aruco marker for", Threshold_no_marker, "consecutive frames.")

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
