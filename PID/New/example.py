import serial
import time
import cv2
from cv2 import aruco
import numpy as np
import math
import queue, threading

# Serial communication with the robot
ser = serial.Serial('COM8', 9600, timeout=1)

# Tolerance for reaching a position
tolerance = 10  # Adjust the tolerance as per your requirements

# ArUco dictionary and parameters
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Camera calibration matrices
camera_matrix = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("G:/sem 7/FYP/New Git/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")

# Region of interest (ROI) coordinates
start_x = 70
start_y = 5
end_x = 716
end_y = 589

# Moving average filter parameters
filter_size = 5
centroid_buffer = []

# Path coordinates
path = [[342, 266], [350, 266], [360, 266], [370, 266], [370, 266],[342, 266], [350, 266], [360, 266], [370, 266], [370, 266] ]

centroid_buffer = []  # Global variable for storing centroid positions




# bufferless VideoCapture
class VideoCapture:
  
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

# Function to send movement command to the robot
def send_command_to_esp32(command):
    ser.write(command.encode())
    response = ser.readline().decode().rstrip()
    if response:
        print(response)
    while ser.readline().decode().rstrip() != 'ACK':
        pass

integral = 0.0

def pi_controller(target, current, kp=0.1, ki=0.01):
    error = target - current
    integral = integral + error
    control = kp * error + ki * integral
    return control


def move_robot_to_coordinates(x, y):
    global centroid_buffer  # Declare centroid_buffer as a global variable

    # Get the current robot position
    robot_x, robot_y = centroid_buffer[-1]  # Assuming centroid_buffer contains the robot's position

    # Calculate the necessary movement commands to reach the given coordinates
    dx = x - robot_x
    dy = pi_controller(y, robot_y)

    # Move diagonally
    if dx < -tolerance and dy < -tolerance:
        command = '9'  # Forward Left
    elif dx > tolerance and dy < -tolerance:
        command = '7'  # Forward Right
    elif dx < -tolerance and dy > tolerance:
        command = '1'  # Backward Left
    elif dx > tolerance and dy > tolerance:
        command = '3'  # Backward Right
    # Move horizontally
    elif dx < -tolerance:
        command = '4'  # Left
    elif dx > tolerance:
        command = '6'  # Right
    else:
        command = ''  # Stay in the same position horizontally

    send_command_to_esp32(command)  # Send the horizontal movement command
   
    # Move vertically
    if dy < -tolerance:
        command = '8'  # Forward
    elif dy > tolerance:
        command = '2'  # Backward
    else:
        command = ''  # Stay in the same position vertically

    if command:
        send_command_to_esp32(command)  # Send the vertical movement command

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
url = "G:/sem 7/FYP/New Git/FYP-Transformer-Robot/output.avi"
# Video capture
#cap = cv2.VideoCapture(url)
cap = VideoCapture(url)
while True:
    
    frame = cap.read()
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

    if len(corners) > 0:
        rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
        rvec = np.array(rvec).reshape((3,))
        R, _ = cv2.Rodrigues(rvec)
        z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

        centroid = np.mean(corners[0][0], axis=0)
        centroid_buffer.append(centroid)
        if len(centroid_buffer) > filter_size:
            centroid_buffer.pop(0)
        print(centroid_buffer[-1])

        if len(path) > 0:
                x, y = path[0]  # Get the next target coordinates from the path
                while True:
                    move_robot_to_coordinates(x, y)
                path.pop(0)  # Remove the visited target from the path

    if cv2.waitKey(1) & 0xFF == ord('q') or len(path) == 0:
        break

if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

cap.release()
