import matplotlib.pyplot as plt

# First set of coordinates
coordinates1 = [(1, 3), (3, 4),(9,4)]

# Extracting x and y values from the first set of coordinates
x1 = [coord[0] for coord in coordinates1]
y1 = [coord[1] for coord in coordinates1]

# Second set of coordinates
coordinates2 = [(2, 5), (4, 6),(5,3)]

# Extracting x and y values from the second set of coordinates
x2 = [coord[0] for coord in coordinates2]
y2 = [coord[1] for coord in coordinates2]

# Plotting the coordinates
plt.plot(x1, y1, 'r', label='Set 1')
plt.plot(x2, y2, 'b', label='Set 2')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Two Sets of Coordinates')

# Adding a legend
plt.legend()

# Displaying the graph
plt.show()
