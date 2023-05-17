import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Read the image from the camera
ret, frame = cap.read()

# Check if the frame was successfully read
if not ret:
    print("Failed to capture frame from camera")
    exit()

# Release the camera capture
cap.release()

# Save the image to a file
filename = "camera_image.jpg"
cv2.imwrite(filename, frame)

print(f"Image saved as {filename}")
