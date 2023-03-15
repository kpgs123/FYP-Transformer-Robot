import matplotlib.pyplot as plt
import numpy as np
# create data
x = [-8.5,8.5,8.5,25.5]
y = [-25.5,-25.5,8.5,8.5]

x1 = np.linspace(25.5, 32.5416, 1000)
y1 = np.sqrt(17**2*2 -(x1 - 8.5)**2)-8.5

x2 = np.linspace(-8.5, 32.5416, 1000)
y2 = -np.sqrt(17**2*2 -(x2 - 8.5)**2)-8.5

x3 = [8.5,-8.5,-8.5,-25.5]
y3 = [42.5,42.5,8.5,8.5]

x4 = np.linspace(8.5, -32.5416, 1000)
y4 = np.sqrt(17**2*2 -(x4 + 8.5)**2) + 25.5

x5 = np.linspace(-32.5416, -25.5, 1000)
y5 = -np.sqrt(17**2*2 -(x5 + 8.5)**2) + 25.5

# create a figure 
fig, ax = plt.subplots()
ax.set_aspect('equal') # Set the aspect ratio to 1:1


# plot data on the axis
ax.plot(x,y, color= 'blue')
ax.plot(x1,y1, color= 'blue')
ax.plot(x2,y2, color= 'blue')
ax.plot(x3,y3, color= 'blue')
ax.plot(x4,y4, color= 'blue')
ax.plot(x5,y5, color= 'blue')

# show the plot
plt.show()