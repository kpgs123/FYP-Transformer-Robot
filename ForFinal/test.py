import maze


pathM=maze.final_path
pathO1 = pathM['O']
pathO = [(y, x) for x, y in pathO1]
#pathO = pathM['O']
pathI1 = pathM['I']
pathI = [(y, x) for x, y in pathI1] 

print(pathO)

print(pathI)
