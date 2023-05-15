import cv2
import time

# MJPEG stream URL with authentication
#http://192.168.0.90/mjpg/video.mjpg?timestamp=1684139240613
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the video capture object was successfully opened
if not cap.isOpened():
    print("Failed to open video capture")
    exit()

# Read frames from the video capture object
while True:
    # Read the next frame from the stream
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Get the timestamp from the HTTP headers
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Do some processing on the frame here
    # ...

    # Display the frame with the timestamp
    #cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000)), 
                #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Frame", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
