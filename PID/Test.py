import cv2
import time

def preview_video(video_path):
    # Open the video file or camera feed
    video = cv2.VideoCapture(video_path)

    # Check if the video source was successfully opened
    if not video.isOpened():
        print("Error opening video source")
        return

    while True:
        # Read a frame from the video source
        ret, frame = video.read()

        # If the frame was not read successfully, the video has ended
        if not ret:
            print("Video ended")
            break

        # Display the frame
        cv2.imshow('Video Preview', frame)

        # Wait for 1 second (1000 milliseconds)
        #time.sleep(.5)

        # Check if the 'q' key was pressed to stop the preview
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video source and close the preview window
    video.release()
    cv2.destroyAllWindows()


url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

preview_video(url)
