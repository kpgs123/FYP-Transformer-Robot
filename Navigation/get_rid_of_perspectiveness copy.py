import cv2 as cv
import numpy as np

# Load the image
img = cv.imread("demo_Environment.jpg")
M = cv.getRotationMatrix2D((225, 375), -0.6, 1.0)
img = cv.warpAffine(img, M, (img.shape[:2][::-1]))
img = img[22:400, 30:726]
#img_size = img.shape[:2]
#img = cv.resize(img, (img_size[1] // 2, img_size[0] // 2), interpolation = cv.INTER_CUBIC)

# Specify the source points
src_points = []

# Define the callback function for selecting the points
def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        src_points.append((x, y))

# Set the callback function for the image
cv.namedWindow('Image')
cv.setMouseCallback('Image', select_points)

while True:
    # Display the image and wait for a key press
    cv.imshow('Image', img)
    key = cv.waitKey(1) & 0xFF
    
    # If the 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

# Calculate the transformation matrix
h, w = 250, 900
dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
M = cv.getPerspectiveTransform(np.float32(src_points), dst_points)

# Transform the image
img_warped = cv.warpPerspective(img, M, (w, h))

# Display the transformed image
cv.imshow('Transformed Image', img_warped)
cv.waitKey(0)