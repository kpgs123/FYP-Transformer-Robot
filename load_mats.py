import numpy as np

cam1 = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/1/camera_matrix.npy")
dis1 = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/1/distortion_coeffs.npy")

cam2 = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/2/camera_matrix.npy")
dis2 = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/2/distortion_coeffs.npy")

print(cam1)
print(dis1)

print(cam2)
print(dis2)