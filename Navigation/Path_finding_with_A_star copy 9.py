import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import keyboard
import serial
import time
import urllib.request


# Load the image
# URL of the IP cam photo stream
url = 'http://192.168.90.1:8080/shot.jpg'

# Read the photo stream from the URL
img_resp = urllib.request.urlopen(url)

# Convert the response to a NumPy array
img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)

# Decode the NumPy array to an OpenCV image
img = cv.imdecode(img_arr, -1)

#img = cv.imread("D:/Git/FYP-Transformer-Robot/Navigation/2.jpg")

#img = cv.resize(img, (img.shape[:2][1] // 2, img.shape[:2][0] // 2), interpolation = cv.INTER_CUBIC)

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
h, w = 400, 400
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
print(hsv[260, 130])


# Define the range of hue, saturation, and value values to keep
lower_threshold = (0, 0, 0)
upper_threshold = (120, 255, 255)

# Threshold the image to create a binary image
binary_image = cv.inRange(hsv, lower_threshold, upper_threshold)

# Invert the binary image
binary_image = cv.bitwise_not(binary_image)

# Remove small white regions from the image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
binary_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

# Apply the binary image as a mask to the original image
masked_image = cv.bitwise_and(img, img, mask=binary_image)

'''for identify the main path'''

# Define lower and upper thresholds for brown
lower_brown = (90, 0, 0)
upper_brown = (120, 90, 180)

# Create binary mask using inRange function
brown_mask = cv.inRange(hsv, lower_brown, upper_brown)

# Apply mask to original image using bitwise_and function
brown_img = cv.bitwise_and(img, img, mask=brown_mask)
brown_mask_bool = brown_mask.astype(np.int32)

# Display the extracted brown pixelsq
cv.imshow("Brown Part of Image", brown_img)
cv.waitKey(0)


'''end'''

# Show the original and masked images
cv.imshow('Original', img)
cv.imshow('Masked', masked_image)
cv.waitKey(0)
cv.destroyAllWindows()

bin_2 = cv.bitwise_not(binary_image)
cv.imshow("Shadows removed", bin_2)
cv.waitKey(0)
img_size = img.shape[:2]
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.bitwise_or(gray_img, bin_2)
#gray_img = cv.resize(gray_img, (gray_img.shape[:2][1] // 2, gray_img.shape[:2][0] // 2), interpolation = cv.INTER_CUBIC)
cv.imshow("gray_img", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()

gray_img = cv.resize(gray_img, (gray_img.shape[:2][1], gray_img.shape[:2][0]), interpolation = cv.INTER_CUBIC)
blured_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow("Blured Image", blured_img)
cv.waitKey(0)
cv.destroyAllWindows()
ret, thresh = cv.threshold(blured_img, 210, 255, cv.THRESH_BINARY)
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

maze = virtualBarrier(2)
maze = np.array(maze)
maze = maze.astype(np.int32)
plt.imshow(maze)
plt.show()

with open("maze.txt", 'w') as file:
   file.writelines(str(list(maze)))

def astar(start, goal, grid):
    """
    Implements the A* algorithm to find the shortest path from start to goal in a 2D grid.
    :param start: a tuple representing the starting position in the grid
    :param goal: a tuple representing the destination position in the grid
    :param grid: a NumPy array representing the 2D grid, where 1 represents a barrier and 0 represents a square that
                 can be moved through
    :return: a tuple of two elements - the first element is the length of the shortest path from start to goal, and the
             second element is the path itself as a list of positions
    """
    # initialize the priority queue
    pq = []
    heapq.heappush(pq, (0, start, [start]))
    # initialize the visited set
    visited = set()

    

    while len(pq) > 0:
        t = 8
        # pop the position with the lowest f-score (i.e., g-score + h-score) from the priority queue
        f, pos, path = heapq.heappop(pq)

        if pos in visited:
            continue
        # if the goal position is reached, return the path
        if pos == goal:
            return (f, path)

        visited.add(pos)
        # expand the position's neighbors and add them to the priority queue
        for neighbor in get_neighbors(pos, grid):
            if neighbor in visited:
                continue
            # calculate the g-score and h-score of the neighbor
            cond = (abs(pos[0] - neighbor[0])) and abs(pos[1] - neighbor[1])
            if brown_mask_bool[pos[0], pos[1]] == 0:
                g = t*100000
            else:
               g = 0
            if not cond:
                g += f + 1*t
            else:
                g += f + math.sqrt(2)*t
                #g = f + 2*t**2

            h = heuristic(neighbor, goal)
            heapq.heappush(pq, (g + h, neighbor, path + [neighbor]))

    # if the goal position is not reached, return None
    return None

def heuristic(pos, goal):
    """
    Calculates the heuristic score between the given position and the goal position. In this implementation, the
    heuristic score is the Manhattan distance between the positions.
    :param pos: a tuple representing the position for which to calculate the heuristic score
    :param goal: a tuple representing the goal position
    :return: the heuristic score between the position and the goal position
    """

    x1, y1 = pos
    x2, y2 = goal

    return abs(x1 - x2) + abs(y1 - y2) #manhattan_distance
    
    #return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) #euclidean_distance

    #return max(abs(x1 - x2), abs(y1 - y2)) #chebyshev_distance

    '''dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy) #diagonal_distance'''

    #return 0 #zero_distance

    

def get_neighbors(pos, grid):
    """
    Returns a list of positions that are adjacent to the given position and are not barriers in the grid.
    :param pos: a tuple representing the position for which to get neighbors
    :param grid: a NumPy array representing the 2D grid
    :return: a list of positions that are adjacent to the given position and are not barriers in the grid
    """
    t = 8


    neighbors = []
    
    for dx, dy in [(1*t, 0*t), (-1*t, 0*t), (0*t, 1*t), (0*t, -1*t), (-1*t, -1*t), (1*t, -1*t), (1*t, 1*t), (-1*t, 1*t)]:
        x, y = pos[0] + dx, pos[1] + dy
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1] or grid[x, y] == 1:
            continue
        neighbors.append((x, y))
    return neighbors

# set the start and goal positions
start = (32, 288)
goal = (272, 72)

# find the shortest path from start to goal using the A* algorithm
print(maze.shape)
path_length, path = astar(start, goal, maze)

# print the results
if path is not None:
    print(f"Shortest path length: {path_length}")
    print(f"Shortest path: {path}")
else:
    print("No path found!")

r,c = maze.shape
with open("solved_maze.txt", 'w') as file:
    for row in range(r):
        s = ""
        for column in range(c):
            square = maze[row][column]
            if (row, column) in path:
                s += '+'
            elif square == 1:
                s += '%'
            else:
                s += '&'
            s += ' '
        file.write(s+'\n')

backtorgb = cv.cvtColor(thresh,cv.COLOR_GRAY2RGB)

new_img = np.array(img)

path_arr = np.zeros([r, c])
for ind in range(len(path) -1):
    y1, x1 = path[ind]
    y2, x2 = path[ind+1]
    cv.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    

plt.imshow(new_img)
plt.show()

orientations = []

for node_index in range(len(path)-1):
   x2, y2 = path[node_index + 1]
   x1, y1 = path[node_index]
   if x2 - x1 == 0:
    if y2 - y1 > 0:
       direction = '4'
    else:
       direction = '6'
   elif y2 - y1 == 0:
    if x2 - x1 > 0:
       direction = '2'
    else:
       direction = '8'
   elif x2 - x1 > 0 and y2 - y1 > 0:
      direction = '9'
   elif x2 - x1 < 0 and y2 - y1 > 0:
      direction = '7'
   elif x2 - x1 < 0 and y2 - y1 < 0:
      direction = '1'
   elif x2 - x1 > 0 and y2 - y1 < 0:
      direction = '3'
    
   orientations.append(direction)

print(orientations)

scale_factor = 12

# Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
ser = serial.Serial('COM4', 9600, timeout=2)

# Define a callback function to handle key presses
def sendNode(oreintation):
    t1 = time.time()
    t2 = time.time()
    while t2 - t1 < 2:
        ser.write(str(oreintation).encode())
        time.sleep(0.2)
        t2 = time.time()

'''for i in orientations:
   sendNode(i)'''

# Keep the program running to allow key presses to be detected
i = 0
while i < len(orientations) - 1:
    sendNode(orientations[i])
    data = ser.readline()
    s = data.decode()
    s = s[:-2]
    if len(s):
        print(s)
    i += 1