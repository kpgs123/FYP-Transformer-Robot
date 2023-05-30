import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import keyboard
import serial
import time
from cv2 import aruco


def nearest_pix_cord(cord):
   t = 5
   return cord - cord % t

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()


#url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
path = np.empty((0, 2), float)
camera_matrix = np.load("D:/Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("D:/Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Define the region of interest (ROI) to crop
start_x = 100  # Starting x-coordinate of the ROI
start_y = 0  # Starting y-coordinate of the ROI
end_x = 700   # Ending x-coordinate of the ROI
end_y = 600   # Ending y-coordinate of the ROI



#cap = cv.VideoCapture(url)
no_marker_count = 0
Threshold_no_marker = 55

fps_limit = 10  # Desired frame rate
frame_interval = 1 / fps_limit  # Time interval between frames

frame = cv.imread("D:/Git/FYP-Transformer-Robot/pic/picture1.jpg")

# Undistort the frame
undistorted_frame = cv.undistort(frame, camera_matrix, dist_coeffs)

# Crop the undistorted frame
cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]
angle = 180
height, width = cropped_frame.shape[:2]
center = (width // 2, height // 2)

rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
img = cv.warpAffine(cropped_frame, rotation_matrix, (width, height))

# Convert the image to the HSV color space
# Apply Gaussian blur
#img = cv.GaussianBlur(img, (9, 9), 0)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)
print(hsv[440, 70])



# Define the range of hue, saturation, and value values to keep
lower_threshold = (0, 0, 0)
upper_threshold = (160, 255, 255)

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
lower_brown = (50, 75, 40)
upper_brown = (130, 255, 255)

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

#gray_img = cv.resize(gray_img, (gray_img.shape[:2][1], gray_img.shape[:2][0]), interpolation = cv.INTER_CUBIC)
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

maze = virtualBarrier(5)
maze = np.array(maze)
maze = maze.astype(np.int32)
np.save("maze.npy", maze)
plt.imshow(maze)
plt.show()

# Load the image (1 represents movable area, 0 represents obstacle area)
image = np.float32(maze) * 255
print(image.shape)

# Create Gaussian kernel
kernel_size = (50, 50)  # Adjust the kernel size for desired thickness
sigma = 1250  # Adjust the sigma value for the spread of the Gaussian
gaussian_kernel = cv.getGaussianKernel(kernel_size[0], sigma) @ cv.getGaussianKernel(kernel_size[1], sigma).T
#box_kernel_img = cv.boxFilter(image, -1, kernel_size)

# Perform 2D convolution with Gaussian kernel
thicker_image = cv.filter2D(image, -1, gaussian_kernel)

# Normalize thicker image to range 0-1
prox_maze = thicker_image


plt.imshow(prox_maze)
plt.show()

def astar(start, goal, grid, prox_grid):
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
        t = 5
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

            # Given coordinates in cm
            x_cm = [8.5,8.5,-8.5,-8.5,8.5]
            y_cm = [25.5,-42.5,-42.5,25.5,25.5]

            # Image and area dimensions
            image_size = 600
            area_width_cm = 220
            area_height_cm = 220

            # Calculate the scaling factor
            scale_x = image_size / area_width_cm
            scale_y = image_size / area_height_cm

            # Pixel offset from the origin
            offset_x = neighbor[0]
            offset_y = neighbor[1]

            # Scale the coordinates from cm to pixels
            x_px = [(int(-x * scale_x) + offset_x) for x in x_cm]
            y_px = [(int(-y * scale_y) + offset_y) for y in y_cm]

            x_set = sorted(set(x_px))
            y_set = sorted(set(y_px))

            cost_for_collision = obstcle_inside_the_shape_o(x_set[0], x_set[1], y_set[0], y_set[1], prox_grid)

            cond = (abs(pos[0] - neighbor[0])) and abs(pos[1] - neighbor[1])
            if brown_mask_bool[pos[0], pos[1]] == 0:
                g = cost_for_collision / 255 + 100
            else:
                g = cost_for_collision / 255
            if not cond:
                g += f + 1*t
            else:
                g += f + math.sqrt(2)*t*10
                #g = f + 2*t**2

            h = heuristic(neighbor, goal)
            heapq.heappush(pq, (g + h, neighbor, path + [neighbor]))

    # if the goal position is not reached, return None
    print(path)
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
    t = 5


    neighbors = []
    
    for dx, dy in [(1*t, 0*t), (-1*t, 0*t), (0*t, 1*t), (0*t, -1*t), (-1*t, -1*t), (1*t, -1*t), (1*t, 1*t), (-1*t, 1*t)]:
        x, y = pos[0] + dx, pos[1] + dy
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1] or grid[x, y] == 1:
            continue
        neighbors.append((x, y))
    return neighbors

def obstcle_inside_the_shape_o(x1, x2, y1, y2, prox_grid):
    if x2 > 599:
       x2 = 599
    if y2 > 599:
       y2 = 599
    count = 0
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            count += prox_grid[x, y]
    return count

# set the start and goal positions
'''start = (50, 205)
goal = (400, 50)'''


frame = cv.imread("D:/Git/FYP-Transformer-Robot/imgesOfRobo/image1.jpg")

# Undistort the frame
undistorted_frame = cv.undistort(frame, camera_matrix, dist_coeffs)

# Crop the undistorted frame
cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

angle = 180
height, width = cropped_frame.shape[:2]
center = (width // 2, height // 2)


rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
cropped_frame = cv.warpAffine(cropped_frame, rotation_matrix, (width, height))

gray = cv.cvtColor(cropped_frame, cv.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)


if len(corners) > 0:
    no_marker_count = 0
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)

    rvec = np.array(rvec).reshape((3,))
    z_rot = rvec[2]
    z_rot_deg = round(math.degrees(z_rot), 2)

    R, _ = cv.Rodrigues(rvec)
    z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

    print(z_rot)

    # Get x and y vectors in marker's coordinate system
    x_axis = np.dot(R, np.array([1, 0, 0]).T)
    y_axis = np.dot(R, np.array([0, 1, 0]).T)

    centroid = np.mean(corners[0][0], axis=0)
    centroid = centroid[::-1]
    start = tuple(map(nearest_pix_cord, map(int, centroid)))
    print(start)

goal = (130, 50) # must provide integer multiplication of t

# find the shortest path from start to goal using the A* algorithm
print(maze.shape)

# print the results
path_length, path = astar(start, goal, maze, prox_maze)
print(f"Shortest path length: {path_length}")
print(f"Shortest path: {path}")
np.save("path.npy", path)
r,c = maze.shape

backtorgb = cv.cvtColor(thresh,cv.COLOR_GRAY2RGB)

new_img = np.array(cropped_frame)

path_arr = np.zeros([r, c])
for ind in range(len(path) -1):
    y1, x1 = path[ind]
    y2, x2 = path[ind+1]
    cv.line(new_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Given coordinates in cm
    x_cm = [8.5,8.5,-8.5,-8.5,8.5]
    y_cm = [25.5,-42.5,-42.5,25.5,25.5]

    # Image and area dimensions
    image_size = 600
    area_width_cm = 220
    area_height_cm = 220

    # Calculate the scaling factor
    scale_x = image_size / area_width_cm
    scale_y = image_size / area_height_cm

    # Pixel offset from the origin
    offset_x = x1
    offset_y = y1

    # Scale the coordinates from cm to pixels
    x_px = [(int(-x * scale_x) + offset_x) for x in x_cm]
    y_px = [(int(-y * scale_y) + offset_y) for y in y_cm]

    # Draw the square on the image
    points = np.array([(x, y) for x, y in zip(x_px, y_px)], np.int32)
    points = points.reshape((-1, 1, 2))
    color = (255, 0, 0)  # Blue color in BGR format
    thickness = 1
    cv.polylines(new_img, [points], isClosed=True, color=color, thickness=thickness)


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
            direction = '8'
        else:
            direction = '2'
    elif x2 - x1 > 0 and y2 - y1 > 0:
        direction = '7'
    elif x2 - x1 < 0 and y2 - y1 > 0:
        direction = '1'
    elif x2 - x1 < 0 and y2 - y1 < 0:
        direction = '3'
    elif x2 - x1 > 0 and y2 - y1 < 0:
        direction = '9'
        
    orientations.append(direction)

print(orientations)

scale_factor = 12

# Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
ser = serial.Serial('COM7', 9600, timeout=2)

# Define a callback function to handle key presses
def sendNode(oreintation):
    #t1 = time.time()
    #t2 = time.time()
    ser.write(str(oreintation).encode())
    time.sleep(0.4)
    #t2 = time.time()

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
    #print("No path found!")

