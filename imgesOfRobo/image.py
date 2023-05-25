import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture("rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1")

# Capture an image
ret, frame = cap.read()


# Create a counter
count = 1

while True:

    # Capture an image
    ret, frame = cap.read()

    # Check if the user pressed the `q` key
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):

        # Save the image
        cv2.imwrite("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/imgesOfRobo/image{0}.jpg".format(count), frame)

        # Increment the counter
        count += 1

        # Continue capturing images
        continue

    # Display the image
    cv2.imshow('frame', frame)

    # Check if the user pressed the `esc` key
    if k == 27:
        break

# Close the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
