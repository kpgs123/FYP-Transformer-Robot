import cv2
import cv2.aruco as aruco
cap = cv2.VideoCapture(0)  # Use the appropriate camera index if not the default one
cv2.namedWindow('Video Feed')
selected_points = []
corner_count = 0
def mouse_callback(event, x, y, flags, param):
    global selected_points, corner_count

    if event == cv2.EVENT_LBUTTONDOWN:
        if corner_count < 4:
            selected_points.append((x, y))
            corner_count += 1
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Video Feed', frame)


cv2.setMouseCallback('Video Feed', mouse_callback)

while True:
    ret, frame = cap.read()
    cv2.imshow('Video Feed', frame)

    if corner_count == 4:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    ret, frame = cap.read()

    # Perform ArUco marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Iterate over detected markers and draw bounding boxes
        for i in range(len(ids)):
            aruco.drawDetectedMarkers(frame, corners)

    # Display the frame with ArUco markers
    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
