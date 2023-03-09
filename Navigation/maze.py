import cv2
import numpy as np

def create_maze(walls):
    # Determine the dimensions of the maze
    min_x, min_y, max_w, max_h = walls[0]
    for x, y, w, h in walls:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_w = max(max_w, x + w)
        max_h = max(max_h, y + h)
    width = max_w - min_x
    height = max_h - min_y

    # Create a matrix to represent the maze
    maze = np.zeros((height, width))

    # Set the cells in the matrix corresponding to the walls
    for x, y, w, h in walls:
        maze[y:y+h, x:x+w] = 1

    return maze

def solve_maze(maze):
    # Find the starting position
    start_row, start_col = np.where(maze == 0)
    start_row = start_row[0]
    start_col = start_col[0]

    # Initialize the stack with the starting position
    stack = [(start_row, start_col, [])]

    # Set of visited cells
    visited = set()

    # Direction vectors for moving up, down, left, and right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Main loop
    while stack:
        # Get the current position and path
        row, col, path = stack.pop()

        # Mark the current position as visited
        visited.add((row, col))

        # Check if we have reached the end of the maze
        if row == 0 or row == maze.shape[0] - 1 or col == 0 or col == maze.shape[1] - 1:
            return path

        # Try moving to each of the four directions
        for direction in directions:
            # Calculate the new position
            new_row = row + direction[0]
            new_col = col + direction[1]

            # Check if the new position is valid and has not been visited
            if 0 <= new_row < maze.shape[0] and 0 <= new_col < maze.shape[1] and maze[new_row, new_col] == 0 and (new_row, new_col) not in visited:
                # Add the new position to the stack with the updated path
                stack.append((new_row, new_col, path + [(row, col)]))

    # If the stack is empty, there is no solution
    return []

# Load the image
image = cv2.imread('maze.jpg')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.resize(gray, (gray[1] // 2, gray[0] // 2), interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray, (5,5), 0)
threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("thresh", threshold)
cv2.waitKey(0)

# Perform edge detection
edges = cv2.Canny(threshold, 50, 150)
cv2.imshow("edges", edges)
cv2.waitKey(0)
# Perform morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours in the image
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
new_img = image.copy()
cv2.drawContours(new_img, contours, -1, (0, 255, 0), 3)
cv2.imshow("contours", new_img)
cv2.waitKey(0)

# Create a list of coordinates for each wall
walls = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    walls.append((x, y, w, h))

print(walls)
new_img = image.copy()
for wall in walls:
    cv2.rectangle(new_img, wall,(0,255,0), 9)
cv2.imshow("walls", new_img)
cv2.waitKey(0)
# Create a representation of the maze using the coordinates
maze = create_maze(walls)

# Solve the maze using the representation
solution = solve_maze(maze)
print(solution)