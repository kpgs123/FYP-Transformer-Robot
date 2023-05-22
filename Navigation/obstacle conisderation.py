import cv2
import numpy as np

def apply_kernel(image, kernel_size, obstacle_distance):
    # Convert the image to float32 for computations
    image_float = image.astype(np.float32)
    
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # Compute the distance transform of the obstacle pixels
    obstacle_dist = obstacle_distance * (1 - image_float / 255.0)
    
    # Apply convolution
    convolved = cv2.filter2D(obstacle_dist, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    # Normalize the convolved values between 0 and 1
    normalized = (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))
    
    return normalized

# Example usage
image = np.array([[1, 0, 0, 1, 1],
                  [1, 1, 0, 1, 1],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

kernel_size = 3
obstacle_distance = 2

result = apply_kernel(image, kernel_size, obstacle_distance)
print(result)
