import cv2
import numpy as np

# Load the image
image = cv2.imread('demo_Environment.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find the contours of the image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty mask
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Loop through the contours and fill the shadowed regions with white
for cnt in contours:
    if cv2.contourArea(cnt) < 107525:
        cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)

# Remove the shadows from the image by setting the shadowed pixels to white
image = cv2.bitwise_and(image, image, mask=mask)

# Save the result
cv2.imshow('shadow_removed.jpg', image)
cv2.waitKey(0)