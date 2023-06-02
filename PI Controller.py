import cv2
from cv2 import aruco
import time
import numpy as np
import math
import serial

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
#url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
path = np.empty((0, 2), float)
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")
# Define the region of interest (ROI) to crop
start_x = 70  # Starting x-coordinate of the ROI
start_y = 5  # Starting y-coordinate of the ROI
end_x = 716   # Ending x-coordinate of the ROI
end_y = 589   # Ending y-coordinate of the ROI

 # Initialize variables for the integral term and previous error
integral_x = 0
integral_y = 0
prev_error_x = 0
prev_error_y = 0


cap = cv2.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames

desired_x = 225
desired_y = 300
desired_pose = (desired_x, desired_y)

# Define the gains for the PI controller
Kp = 0.5  # Proportional gain
Ki = 0.1  # Integral gain
dt = 0.1  #time step

# Define the error margin in pixels
error_margin = 4


try:
    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()

        # No more frames in video, break out of loop
        if not ret:
            break

        current_time = time.time()
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
                cv2.putText(cropped_frame, f"position: {centroid}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

                # Loop until the robot reaches the desired pose
            
                current_pose = (centroid[0], centroid[1])
                while True:
                    # Calculate the error vector
                    error_vector = (desired_pose[0] - current_pose[0], desired_pose[1] - current_pose[1])
                    print(error_vector)

                    # Check if the error distance is within the error margin
                    if abs(error_vector[0]) <= error_margin and abs(error_vector[1]) <= error_margin:
                        print("Robot reached the desired pose.")
                        break
                    else:
                
                        # Update the integral term
                        integral_x += error_vector[0] * dt
                        integral_y += error_vector[1] * dt

                        # Calculate the control signal
                        control_x = Kp * error_vector[0] + Ki * integral_x + Kp * (error_vector[0] - prev_error_x)
                        control_y = Kp * error_vector[1] + Ki * integral_y + Kp * (error_vector[1] - prev_error_y)

                        orientations = []
                        # control signal to control the robot's movement
                        if control_x >= error_margin and control_x > 0:
                            orientations.append(8)
                        elif control_x >= error_margin and control_x < 0:
                            orientations.append(2)
                        elif control_y >= error_margin and control_y > 0:
                            orientations.append(6) 
                        elif control_y >= error_margin and control_y < 0:
                            orientations.append(4)  

                    # Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
                    ser = serial.Serial('COM9', 9600, timeout=20)

                    # Define a callback function to handle key presses
                    def sendNode(oreintation):
                        t1 = time.time()
                        t2 = time.time()
                        while t2 - t1 < 2:
                            ser.write(str(oreintation).encode())
                            time.sleep(0.5)
                            t2 = time.time()

                        '''for i in orientations:
                        sendNode(i)'''

                        # Keep the program running to allow key presses to be detected
                    i = 0
                    while i < len(orientations) - 1:
                        sendNode(orientations[i])
                        data = ser.readline()
                        s = data.decode()
                        s = s[:-2]
                        if len(s):
                            print(s)
                        i += 1
                     

                        # Update the current pose of the robot
                        ret, frame = cap.read()
                        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                        cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]
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
                        frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
                        cv2.imshow('frame', frame_markers)
                        # Loop until the robot reaches the desired pose
            
                        current_pose = (centroid[0], centroid[1])
                        # Here, you would typically update the current pose based on sensor feedback

                        # Store the current error for the next iteration
                        prev_error_x = error_vector[0]
                        prev_error_y = error_vector[1]


                path = np.append(path , np.array([centroid]), axis=0)

                frame_markers = aruco.drawDetectedMarkers(cropped_frame.copy(), corners, ids)
                cv2.imshow('frame', frame_markers)

                   

            else:
                no_marker_count += 1
                if no_marker_count >= Threshold_no_marker:
                    print("Cannot find aruco marker for", Threshold_no_marker, "consecutive frames.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
        cv2.destroyWindow('frame')

    cap.release()
    print(path)

except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()







