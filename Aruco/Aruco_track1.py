import cv2
import numpy as np

# Define the Aruco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

# Initialize the video capture device
cap = cv2.VideoCapture('G:/sem 7/FYP/Git/FYP-Transformer-Robot/FYP Videos/2_res_Trim.mp4')

# Define the empty matrix to store the tracked path
path = np.empty((0, 2), dtype=np.float32)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the Aruco markers in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Draw the detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Get the centroid of the first detected marker
        centroid = np.mean(corners[0][0], axis=0)
        print(centroid)

        # Append the centroid coordinates to the path matrix
        path = np.append(path, np.array([[centroid[0], centroid[1]]]), axis=0)

        # Draw a line connecting the previous and current centroid
        for i in range(1, len(path)):
            cv2.line(frame, (int(path[i-1][0]), int(path[i-1][1])),
                     (int(path[i][0]), int(path[i][1])), (0, 255, 0), thickness=2)

    # Display the frame
    cv2.imshow("frame", frame)

    # Exit if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the path matrix
print(path)
