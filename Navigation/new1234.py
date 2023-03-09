import heapq
import numpy as np
import math

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
            if not cond:
                g = f + 1
            else:
                g = f + math.sqrt(2)
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
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def get_neighbors(pos, grid):
    """
    Returns a list of positions that are adjacent to the given position and are not barriers in the grid.
    :param pos: a tuple representing the position for which to get neighbors
    :param grid: a NumPy array representing the 2D grid
    :return: a list of positions that are adjacent to the given position and are not barriers in the grid
    """
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1] or grid[x, y] == 1:
            continue
        neighbors.append((x, y))
    return neighbors



import numpy as np

# create a 2D NumPy array representing the grid
# 0 represents a square that can be moved through, and 1 represents a barrier

with open("maze123.txt", 'r') as file:
    s = file.read()
    grid = np.array(eval(s))
    grid = grid.astype(np.int32)

'''grid = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])'''

# set the start and goal positions
start = (0, 0)
goal = (24, 24)

# find the shortest path from start to goal using the A* algorithm
path_length, path = astar(start, goal, grid)

# print the results
if path is not None:
    print(f"Shortest path length: {path_length}")
    print(f"Shortest path: {path}")
else:
    print("No path found!")

r,c = grid.shape
with open("solved_maze.txt", 'w') as file:
    for row in range(r):
        s = ""
        for column in range(c):
            square = grid[row][column]
            if (row, column) in path:
                s += '+'
            elif square == 1:
                s += '%'
            else:
                s += '&'
            s += ' '
        file.write(s+'\n')