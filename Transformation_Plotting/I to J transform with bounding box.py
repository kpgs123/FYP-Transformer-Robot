import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

# create data
x = [8.5, -8.5, -8.5, 8.5]
y = [25.5, 25.5, -42.5, -42.5]

x1 = np.linspace(8.5, -32.5416, 1000)
y1 = np.sqrt(17**2*2 - (x1 + 8.5)**2) + 8.5

x2 = np.linspace(-25.5, -32.5416, 1000)
y2 = -np.sqrt(17**2*2 - (x2 + 8.5)**2) + 8.5

x3 = np.linspace(-25.5, -32.5416, 1000)
y3 = np.sqrt(17**2*2 - (x3 + 8.5)**2) - 25.5

x4 = np.linspace(8.5, -32.5416, 1000)
y4 = -np.sqrt(17**2*2 - (x4 + 8.5)**2) - 25.5

fig, ax = plt.subplots()
ax.set_aspect('equal')  # Set the aspect ratio to 1:1

coordinates = []
for i in range(len(x) - 1):
    coordinates.append((x[i], y[i]))
for j in range(len(x1) - 1):
    coordinates.append((x1[j], y1[j]))
for k in range(len(x2) - 1):
    coordinates.append((x2[k], y2[k]))
for l in range(len(x3) - 1):
    coordinates.append((x3[l], y3[l]))
for m in range(len(x4) - 1):
    coordinates.append((x4[m], y4[m]))
print(coordinates)

circle = Circle((0, 0), radius=1, edgecolor='red', facecolor='red')

# Add the circle to the axes
ax.add_artist(circle)

# Draw a rectangle around the shape
min_x = min([coord[0] for coord in coordinates])
max_x = max([coord[0] for coord in coordinates])
min_y = min([coord[1] for coord in coordinates])
max_y = max([coord[1] for coord in coordinates])
rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, edgecolor='green', facecolor='none')
ax.add_artist(rectangle)

# Print the coordinates of the corners
corners = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
print("Coordinates of the corners:")
for corner in corners:
    print(corner)

# Set the axis limits
ax.set_xlim(-50, 50)
ax.set_ylim(-60, 60)

# plot data on the axis
ax.plot(x, y, color='blue')
ax.plot(x1, y1, color='blue')
ax.plot(x2, y2, color='blue')
ax.plot(x3, y3, color='blue')
ax.plot(x4, y4, color='blue')

# show the plot
plt.show()
