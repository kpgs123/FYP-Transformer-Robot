### Overlay marker ID and video
import cv2
from cv2 import aruco
import time
import numpy as np

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
url= 'http://10.10.2.198:8080/video'

# Define the empty matrix to store the tracked path
path = np.empty((0, 2), dtype=np.float32)

cap = cv2.VideoCapture('G:/sem 7/FYP/Git/FYP-Transformer-Robot/FYP Videos/1_res.mp4')
#cap = cv2.VideoCapture(url)
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        #cv2.imshow('frame2',gray)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if ids is not None:
            
            #Draw detected marker on frame
            cv2.aruco.drawDetectedMarkers(frame,corners,ids)

            #get centroid of marker
            centroid = np.mean(corners[0][0], axis=0)

            #Append the centroid to path matrix
            path = np.append(path,np.array([centroid[0],centroid[1]]),axis=0)

             # Draw a line connecting the previous and current centroid
            for i in range(1, len(path)):
                cv2.line(frame, (int(path[i-1][0]), int(path[i-1][1])),
                    (int(path[i][0]), int(path[i][1])), (0, 255, 0), thickness=2)

        # Display the frame
        cv2.imshow("frame", frame)



        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('frame')
    cap.release()
    print(path)
except KeyboardInterrupt:
    cv2.destroyWindow('frame')
    cap.release()