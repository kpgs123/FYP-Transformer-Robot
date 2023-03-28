### Overlay marker ID and video
import cv2
from cv2 import aruco
import time

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url= 'http://10.10.2.198:8080/video'

#cap = cv2.VideoCapture('G:/sem 7/FYP/Git/FYP-Transformer-Robot/FYP Videos/1_res.mp4')
cap = cv2.VideoCapture(url)
try:
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('frame2',gray)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow('frame', frame_markers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('frame')
    cap.release()
except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()