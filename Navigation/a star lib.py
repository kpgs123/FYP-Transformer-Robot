from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

with open("maze123.txt", 'r') as file:
    s = file.read()
    matrix = eval(s)

grid = Grid(matrix=matrix)

start = grid.node(0, 0)
end = grid.node(49, 49)

finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
path, runs = finder.find_path(start, end, grid)

print(path)

with open("solved_maze.txt", 'w') as file:
    file.write('operations:' + str(runs) + 'path length:' + str(len(path)))
    file.write('\n')
    file.write(grid.grid_str(path=path, start=start, end=end))