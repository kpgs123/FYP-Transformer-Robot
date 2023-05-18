import cv2
import numpy as np

x = [8.5, 8.5, -25.5, -25.5, 8.5]
y = [8.5, -25.5, -25.5, 8.5, 8.5]

origin_x = 50  # X-coordinate of the desired origin
origin_y = 60  # Y-coordinate of the desired origin

# Calculate the coordinates for the square
left = int(min(x)) + origin_x
top = int(min(y)) + origin_y
width = int(max(x) - min(x))
height = int(max(y) - min(y))
right = left + width
bottom = top + height

# Create a black image
image_size = 400
image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# Draw the square on the image
start_point = (left, top)
end_point = (right, bottom)
color = (255, 0, 0)  # Blue color in BGR format
thickness = 2
cv2.rectangle(image, start_point, end_point, color, thickness)

# Display the image
cv2.imshow("Square", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
