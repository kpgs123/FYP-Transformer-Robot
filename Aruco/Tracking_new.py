import cv2
from cv2 import aruco
import time
import numpy as np

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url = 0
path = np.empty((0, 2), float)

cap = cv2.VideoCapture('G:/sem 7/FYP/Git/FYP-Transformer-Robot/FYP Videos/1_res.mp4')
#cap = cv2.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55

try:
    while True:
        ret, frame = cap.read()

        # No more frames in video, break out of loop
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            no_marker_count = 0
            centroid = np.mean(corners[0][0], axis=0)
            path = np.append(path , np.array([centroid]), axis=0)

            frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            cv2.imshow('frame', frame_markers)
        else:
            no_marker_count += 1
            if no_marker_count >= Threshold_no_marker:
                print("Cannot find aruco marker for" , Threshold_no_marker, "consecutive frames.")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
        cv2.destroyWindow('frame')

    cap.release()
    print(path)

except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()
