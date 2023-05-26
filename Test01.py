import cv2
import numpy as np
import matplotlib.pyplot as plt




# Read the image
img = np.load("D:/Git/FYP-Transformer-Robot/maze.npy")

# Print the image
cv2.imshow("img", img)
