import numpy as np

# Create an empty NumPy array
coordinates = np.empty((0, 2), float)

# Add a 2D coordinate to the array
x1 = 1.0
y1 = 2.0
coordinates = np.append(coordinates, np.array([[x1, y1]]), axis=0)

# Add another 2D coordinate to the array
x2 = 3.0
y2 = 4.0
coordinates = np.append(coordinates, np.array([[x2, y2]]), axis=0)

# Print the final array
print(coordinates)
