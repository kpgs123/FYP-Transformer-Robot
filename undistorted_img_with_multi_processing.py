import cv2
import numpy as np
import multiprocessing

def undistort_image(distorted_image, camera_matrix, distortion_coeffs, output):
    # Perform the undistortion
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coeffs)
    # Save the undistorted image
    output.put(undistorted_image)

# Load the distorted image
distorted_image = cv2.imread('/home/geethaka/Documents/Git/FYP-Transformer-Robot/distorted_image.jpg')

# Define the camera matrix and distortion coefficients
camera_matrix = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/2/camera_matrix.npy")

distortion_coeffs = np.load("/home/geethaka/Documents/Git/FYP-Transformer-Robot/2/distortion_coeffs.npy")

distorted_image = cv2.resize(distorted_image, (100, 75))

# Get the number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Create a multiprocessing queue for communication between processes
output = multiprocessing.Queue()

# Create a list to hold the processes
processes = []

# Split the image into smaller regions for parallel processing
regions = np.array_split(distorted_image, num_cores, axis=1)

# Iterate over the regions and create a process for each
for region in regions:
    process = multiprocessing.Process(target=undistort_image, args=(region, camera_matrix, distortion_coeffs, output))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

# Get the undistorted images from the queue
undistorted_images = [output.get() for _ in range(num_cores)]

# Concatenate the undistorted images back together
undistorted_image = np.concatenate(undistorted_images, axis=1)
cv2.imwrite('undistorted_image.jpg', undistorted_image)

# Display the undistorted image
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
