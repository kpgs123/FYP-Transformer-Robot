import cv2
from cv2 import aruco
import time
import numpy as np
import math

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
#url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
path = np.empty((0, 2), float)
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")
print(camera_matrix)
print(dist_coeffs)

cap = cv2.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames

try:
    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()

        # No more frames in video, break out of loop
        if not ret:
            break

        current_time = time.time()

        # Check if the frame should be processed based on the frame interval
        if current_time - last_frame_time >= frame_interval:
            last_frame_time = current_time

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
                cv2.putText(frame, f"Z Rotation: {z_rot}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)

                # Get x and y vectors in marker's coordinate system
                x_axis = np.dot(R, np.array([1, 0, 0]).T)
                y_axis = np.dot(R, np.array([0, 1, 0]).T)

                centroid = np.mean(corners[0][0], axis=0)
                path = np.append(path , np.array([centroid]), axis=0)

                frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
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