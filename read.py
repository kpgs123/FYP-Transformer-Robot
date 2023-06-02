import numpy as np




try:
    # Assuming the data is stored in a file called 'data.txt'
    data = np.genfromtxt('G:/sem 7/FYP/New Git/FYP-Transformer-Robot/maze.npy', delimiter=',')
    print(data)
except ValueError as e:
    print("An error occurred while reading the file:", e)

