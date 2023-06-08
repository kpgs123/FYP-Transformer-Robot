import cv2
import os

# Create a folder to save the captured pictures
folder_name = "picFinal6"
os.makedirs(folder_name, exist_ok=True)

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"

# Open the webcam
cap = cv2.VideoCapture(url)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open the webcam")
    exit()

# Capture and save the specified number of pictures
num_pictures = 10
picture_count = 0

while picture_count < num_pictures:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Failed to read the frame")
        break

    # Display the frame
    cv2.imshow("Frame", frame)

    # Save the frame as an image
    picture_count += 1
    image_path = os.path.join(folder_name, f"picture{picture_count}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Picture {picture_count} saved successfully!")

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()