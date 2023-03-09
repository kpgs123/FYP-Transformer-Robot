import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load the image
img = cv.imread("NEW_Environment_1.png")
#img = cv.resize(img, (img.shape[:2][1] * 2, img.shape[:2][0] * 2), interpolation = cv.INTER_CUBIC)

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
h, w = 320, 600
dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
M = cv.getPerspectiveTransform(np.float32(src_points), dst_points)

# Transform the image
img = cv.warpPerspective(img, M, (w, h))

# Display the transformed image
cv.imshow('Transformed Image', img)
cv.waitKey(0)

# Convert the image to the HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)

# Define the range of hue, saturation, and value values to keep
lower_threshold = (20, 0, 60)
upper_threshold = (190, 235, 255)

# Threshold the image to create a binary image
binary_image = cv.inRange(hsv, lower_threshold, upper_threshold)

# Invert the binary image
binary_image = cv.bitwise_not(binary_image)

# Remove small white regions from the image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
binary_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

# Apply the binary image as a mask to the original image
masked_image = cv.bitwise_and(img, img, mask=binary_image)


# Show the original and masked images
cv.imshow('Original', img)
cv.imshow('Masked', masked_image)
cv.waitKey(0)
cv.destroyAllWindows()

bin_2 = cv.bitwise_not(binary_image)
print(bin_2)
cv.imshow("Shadows removed", bin_2)
cv.waitKey(0)
img_size = img.shape[:2]
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.bitwise_or(gray_img, bin_2)
#gray_img = cv.resize(gray_img, (gray_img.shape[:2][1] // 2, gray_img.shape[:2][0] // 2), interpolation = cv.INTER_CUBIC)
cv.imshow("gray_img", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()

gray_img = cv.resize(gray_img, (gray_img.shape[:2][1] // 2, gray_img.shape[:2][0] // 2), interpolation = cv.INTER_CUBIC)
blured_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow("Blured Image", blured_img)
cv.waitKey(0)
cv.destroyAllWindows()
ret, thresh = cv.threshold(blured_img, 190, 255, cv.THRESH_BINARY)
plt.imshow(thresh)
plt.show()

def bool_mat(mat):
    new_mat = np.zeros(mat.shape, dtype=bool)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            new_mat[i, j] = not bool(mat[i, j])
    return new_mat

maze  = bool_mat(thresh)
r, c = maze.shape

def virtualBarrier(t):
  maze_with_barries = maze.copy()
  for i in range(r):
    for j in range(c):
      if maze[i, j] == 1:
        for ti in range(1, t):
          if i-t > 0:
            maze_with_barries[i-t, j] = 1
          if i+t < r:
            maze_with_barries[i+t, j] = 1
          if j-t > 0:
            maze_with_barries[i, j-t] = 1
          if j+t < c:
            maze_with_barries[i, j+t] = 1
          if i-t > 0 and j-t > 0:
            maze_with_barries[i-t, j-t] = 1
          if i+t < r and j+t < c:
            maze_with_barries[i+1, j+t] = 1
          if j-t > 0 and  i+t < r:
            maze_with_barries[i+t, j-t] = 1
          if j+t < c and i-t > 0:
            maze_with_barries[i-t, j+t] = 1
  return maze_with_barries

maze = virtualBarrier(3)
print(maze)
plt.imshow(maze)
plt.show()

np.savetxt("maze.txt", maze, delimiter=',')

start = (80, 25)
end = (10, 275)

m = np.zeros(maze.shape)

m[start] = 1

def makeStep(k):
  for i in range(r):
    for j in range(c):
      if m[i, j] == k:
        if i-1 > 0 and m[i-1, j] == 0 and maze[i-1, j] == 0:
          m[i-1, j] = k + 1
        if i+1 < r and m[i+1, j] == 0 and maze[i+1, j] == 0:
          m[i+1][j] = k + 1
        if j-1 > 0 and m[i, j-1] == 0 and maze[i, j-1] == 0:
          m[i, j-1] = k + 1
        if j+1 < c and m[i, j+1] == 0 and maze[i, j+1] == 0:
          m[i, j+1] = k + 1

k = 0
while m[end] == 0:
    k += 1
    makeStep(k)

print(m)

i, j = end
k = m[i, j]
path = [(i,j)]
while k > 1:
  if i > 0 and m[i - 1, j] == k-1:
    i, j = i-1, j
    path.append((i, j))
    k-=1
  elif j > 0 and m[i, j - 1] == k-1:
    i, j = i, j-1
    path.append((i, j))
    k-=1
  elif i < r - 1 and m[i + 1, j] == k-1:
    i, j = i+1, j
    path.append((i, j))
    k-=1
  elif j < c - 1 and m[i, j + 1] == k-1:
    i, j = i, j+1
    path.append((i, j))
    k-=1

path = path[::-1]
print(path)

backtorgb = cv.cvtColor(thresh,cv.COLOR_GRAY2RGB)


path_arr = np.zeros([r, c])
for cord in path:
    path_arr[cord] = 1
    backtorgb[cord] = (0, 255, 0)
    

np.savetxt("solved_maze.txt", maze, delimiter=',')

plt.imshow(backtorgb)
plt.show()