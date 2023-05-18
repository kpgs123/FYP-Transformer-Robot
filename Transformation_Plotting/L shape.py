import matplotlib.pyplot as plt
from matplotlib.patches import Circle
x = [8.5,8.5,25.5,25.5,-8.5,-8.5,8.5]
y = [25.5,-8.5,-8.5,-25.5,-25.5,25.5,25.5]

coordinates=[]
for i in range(len(x)-1):
    coordinates.append((x[i],y[i]))
    
print(coordinates)

fig, ax = plt.subplots()
ax.set_aspect('equal') # Set the aspect ratio to 1:1

circle = Circle((0, 0), radius=1, edgecolor='red', facecolor='red')

# Add the circle to the axes
ax.add_artist(circle)

# Set the axis limits
ax.set_xlim(-50, 50)
ax.set_ylim(-60, 60)

ax.plot(x,y, color = 'blue')
plt.show()