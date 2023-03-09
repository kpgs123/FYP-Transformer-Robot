import cv2

# Load the image
image = cv2.imread("demo_Environment.jpg")
image = image[45:375, 100:715]

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

# Define the range of hue, saturation, and value values to keep
lower_threshold = (0, 0, 0)
upper_threshold = (180, 255, 150)

# Threshold the image to create a binary image
binary_image = cv2.inRange(hsv, lower_threshold, upper_threshold)

# Invert the binary image
binary_image = cv2.bitwise_not(binary_image)

# Remove small white regions from the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Apply the binary image as a mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=binary_image)

# Show the original and masked images
cv2.imshow('Original', image)
cv2.imshow('Masked', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()