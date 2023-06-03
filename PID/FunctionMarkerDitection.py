import cv2
from cv2 import aruco
import numpy as np
import math
import time



def process_video(url):
    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
    dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

    # Define the region of interest (ROI) to crop
    start_x = 70  # Starting x-coordinate of the ROI
    start_y = 5  # Starting y-coordinate of the ROI
    end_x = 716   # Ending x-coordinate of the ROI
    end_y = 589   # Ending y-coordinate of the ROI
    
    cap = cv2.VideoCapture(url)
    no_marker_count = 0
    Threshold_no_marker = 55

    fps_limit = 10  # Desired frame rate
    frame_interval = 1 / fps_limit  # Time interval between frames

    path = []

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

                    centroid = np.mean(corners[0][0], axis=0)
                    path.append(centroid)
                    break

                else:
                    no_marker_count += 1
                    if no_marker_count >= Threshold_no_marker:
                        print("Cannot find aruco marker for", Threshold_no_marker, "consecutive frames.")

        cap.release()
        return path

    except KeyboardInterrupt:
        cap.release()
url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

for i in range(80):
    x= process_video(url)
    print (x)
    