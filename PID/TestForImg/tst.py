import cv2

def capture_frame_from_video(video_source):
    # Open the video source (webcam)
    video = cv2.VideoCapture(video_source)

    # Read a frame from the video source
    ret, frame = video.read()

    # Release the video capture object
    video.release()

    # Return the captured frame
    return frame

for i in range(15):
    captured_frame = capture_frame_from_video(0)

    # Display the captured frame
    cv2.imshow('Frame', captured_frame)
    cv2.waitKey(0)

# Release OpenCV window
cv2.destroyAllWindows()

