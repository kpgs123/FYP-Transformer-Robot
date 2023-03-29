### Overlay marker ID and video
import cv2
from cv2 import aruco
import time
import numpy as np

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url= 'http://10.10.2.198:8080/video'
path = np.empty((0, 2), float)

cap = cv2.VideoCapture('G:/sem 7/FYP/Git/FYP-Transformer-Robot/FYP Videos/2_res_Trim.mp4')
#cap = cv2.VideoCapture(url)
try:
    while True:
        ret, frame = cap.read()

        # No more frames in video, break out of loop
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('frame2',gray)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        #print(corners)
        centroid = np.mean(corners[0][0], axis=0)
        #print(centroid)
        path = np.append(path , np.array([centroid]), axis=0)



        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow('frame', frame_markers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('frame')
    cap.release()
    print(path)
except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()