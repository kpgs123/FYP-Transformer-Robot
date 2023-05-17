import cv2

# Define the video capture object
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
video_capture = cv2.VideoCapture(url)  # Use 0 for the default camera

# Define the video codec and create a VideoWriter object
# You can change the filename, codec, frame rate, and resolution as needed
filename = 'video_output.mp4'
codec = cv2.VideoWriter_fourcc(*'mp4v')
frame_rate = 30.0
resolution = (800, 600)
video_writer = cv2.VideoWriter(filename, codec, frame_rate, resolution)

# Start capturing and recording
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Check if the video capture was successful
    if not ret:
        print("Failed to capture video")
        break

    # Write the captured frame to the video file
    video_writer.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
video_capture.release()
video_writer.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
