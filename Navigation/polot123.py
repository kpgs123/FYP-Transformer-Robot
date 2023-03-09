import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-8.5, 32.6, 1000)
y = -np.sqrt(17 ** 2 * 2 - (x - 8.5) ** 2) - 8.5

fig, ax = plt.subplots()
plt.plot(x, y)
ax.set_aspect("equal")
plt.show()