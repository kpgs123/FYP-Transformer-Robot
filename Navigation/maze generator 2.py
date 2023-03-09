import random

def generate_maze(width, height):
    # Initialize the maze with all walls (0's)
    maze = [[0 for _ in range(width)] for _ in range(height)]

    # Start the maze generation from the top-left corner
    generate_maze_recursive(maze, 0, 0)

    # Add a border around the maze
    maze = add_border(maze)

    return maze

def generate_maze_recursive(maze, x, y):
    # Mark the current cell as visited
    maze[y][x] = 1

    # Define the order in which we will try to move in each direction
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    random.shuffle(directions)

    # Try to move in each direction
    for dx, dy in directions:
        # Calculate the new position
        new_x, new_y = x + dx, y + dy

        # Check if the new position is inside the maze and unvisited
        if (0 <= new_x < len(maze[0]) and
            0 <= new_y < len(maze) and
            maze[new_y][new_x] == 0):
            # Break down the wall between the current cell and the new cell
            if dx == 1:
                maze[y][x+1] = 1
            elif dx == -1:
                maze[y][x-1] = 1
            elif dy == 1:
                maze[y+1][x] = 1
            else:
                maze[y-1][x] = 1

            # Recursively generate the maze from the new position
            generate_maze_recursive(maze, new_x, new_y)

def add_border(maze):
    # Add a row of 0's at the top and bottom of the maze
    maze = [[0] * len(maze[0])] + maze + [[0] * len(maze[0])]

    # Add a column of 0's at the left and right of the maze
    for i in range(len(maze)):
        maze[i] = [0] + maze[i] + [0]

    # Set the start and end nodes
    maze[1][1] = 1
    maze[-2][-2] = 1

    return maze




maze = generate_maze(40, 40)
for row in maze:
    print(row)
