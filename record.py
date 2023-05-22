import cv2

# Open the video capture device (0 is usually the default webcam)
cap = cv2.VideoCapture("rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1")

# Check if the video capture device was successfully opened
if not cap.isOpened():
    print("Failed to open video capture device")
    exit()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = "output.avi"
output_size = (800, 600)
fps = 15
out = cv2.VideoWriter(output_file, fourcc, fps, output_size)

# Start recording
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Write the frame to the video file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Check if the 'q' key was pressed to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
