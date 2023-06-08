import serial
import time
import cv2
from cv2 import aruco
import numpy as np
import math
import queue, threading
import matplotlib.pyplot as plt
import maze3


# Serial communication with the robot
ser = serial.Serial('COM4', 9600, timeout=1)

# Tolerance for reaching a position
tolerance = 12  # Adjust the tolerance as per your requirements

# ArUco dictionary and parameters
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Camera calibration matrices
camera_matrix = np.load("E:/sem 7-------------/Final Year Design Project/final/FYP-Transformer-Robot/CaliFinal/camera_matrix.npy")
dist_coeffs = np.load("E:/sem 7-------------/Final Year Design Project/final/FYP-Transformer-Robot/CaliFinal/distortion_coeffs.npy")


start_x = 100  # Starting x-coordinate of the ROI
start_y = 0  # Starting y-coordinate of the ROI
end_x = 700   # Ending x-coordinate of the ROI
end_y = 600   # Ending y-coordinate of the ROI


# Moving average filter parameters
filter_size = 3
centroid_buffer = []

# Path coordinates
#path = [[342, 266], [350, 266], [360, 266], [370, 266], [370, 266],[342, 266], [350, 266], [360, 266], [370, 266], [370, 266] ]
#path = [[285, 186], [300, 186], [320, 186]]



pathM=maze3.final_path
pathO1 = pathM['O']
pathO = [(y, x) for x, y in pathO1]
pathT = pathO.copy()
#pathO = pathM['O']
only_O = False
try:
    pathI1 = pathM['I']
    pathI = [(y, x) for x, y in pathI1]
    pathT.extend(pathI)
except:
    only_O = True
# Extracting x and y values from the first set of coordinates
x1 = [coord[0] for coord in pathT]
y1 = [coord[1] for coord in pathT]

centroid_buffer = []  # Global variable for storing centroid positions

correct_orientation = False


class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        ret, frame = self.q.get()
        return ret, frame

def keep_robot_fixed_orientation(z_rot):
    # Check if the current rotation angle is within the desired range
    if (z_rot < 171) and (z_rot>=0):
        command = 'c'  # Rotate clockwise
    elif (z_rot > -171) and (z_rot<0):
        command = 'a'  # Rotate anticlockwise
    else:
        command = ''  # Stay in the same orientation

    if command:
        send_command_to_esp32(command)  # Send the vertical movement command
    # Return True if the robot is in the correct orientation, False otherwise
    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
            if rvec.size != 3:
                continue
            rvec = np.array(rvec).reshape((3,))
            R, _ = cv2.Rodrigues(rvec)
            z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)
            if z_rot<0:
                z_rot=z_rot+180
            else:
                z_rot= z_rot-180
        if ret:
            break
    
    return z_rot >= 171 or z_rot <= -171 

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

x_flag= False
y_flag=False

tracked = []
def move_robot_to_coordinates(x, y):
    global centroid_buffer  # Declare centroid_buffer as a global variable

    # Get the current robot position
    robot_x, robot_y = centroid_buffer[-1]  # Assuming centroid_buffer contains the robot's position

    # Calculate the necessary movement commands to reach the given coordinates
    dx = x - robot_x
    #dy = pi_controller(y, robot_y)
    dy = y - robot_y

    # Move diagonally
    '''if dx < -tolerance and dy < -tolerance:
        command = '7'  # Forward Left
    elif dx > tolerance and dy < -tolerance:
        command = '9'  # Forward Right
    elif dx < -tolerance and dy > tolerance:
        command = '3'  # Backward Left
    elif dx > tolerance and dy > tolerance:
        command = '1'  # Backward Right'''
    # Move horizontally
    if dx < -tolerance:
        command = '6'  # Right
    elif dx > tolerance:
        command = '4'  # 
    else:
        command = ''  # Stay in the same position horizontally
        global x_flag
        x_flag = True
    if command:
        send_command_to_esp32(command)  # Send the vertical movement command
    
    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
            if rvec.size != 3:
                continue
            rvec = np.array(rvec).reshape((3,))
            R, _ = cv2.Rodrigues(rvec)
            centroid = np.mean(corners[0][0], axis=0)
            tracked.append(centroid)
            if ret:
                break

    robot_x, robot_y = centroid
    #dy = pi_controller(y, robot_y)
    dy = y - robot_y
    # Move vertically
    if dy < -tolerance:
        command = '2'  # Back
    elif dy > tolerance:
        command = '8'  # 
    else:
        command = ''  # Stay in the same position vertically
        global y_flag
        y_flag = True
    if command:
        send_command_to_esp32(command)  # Send the vertical movement command

url = "rtsp://root:abcd@192.168.0.90/axis-media/media.amp?camera=1"
cap = VideoCapture(url)

while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

    if len(corners) > 0:
        rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
        if rvec.size != 3:
            continue
        rvec = np.array(rvec).reshape((3,))
        R, _ = cv2.Rodrigues(rvec)
        z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)
        if z_rot<0:
            z_rot=z_rot+180
        else:
            z_rot= z_rot-180
        centroid = np.mean(corners[0][0], axis=0)
        centroid_buffer.append(centroid)
        tracked.append(centroid)
        if len(centroid_buffer) > filter_size:
            centroid_buffer.pop(0)
        print(centroid_buffer[-1])

        if len(pathO) > 0:
                x, y = pathO[0]  # Get the next target coordinates from the path

                while not correct_orientation:
                    if not correct_orientation:
                        correct_orientation = keep_robot_fixed_orientation(z_rot)

                    # Update the current robot position and rotation
                    robot_x, robot_y = centroid_buffer[-1]
                    z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)
                    if z_rot<0:
                        z_rot=z_rot+180
                    else:
                        z_rot= z_rot-180  
                correct_orientation = False   
                move_robot_to_coordinates(x, y)
                        # Keep moving the robot until it reaches the desired position and orientation
                if x_flag == True and y_flag==True: 
                    pathO.pop(0)  # Remove the visited target from the path
                    x_flag=False
                    y_flag=False

    '''else:
        send_command_to_esp32('a')
        send_command_to_esp32('c')'''
        

    if cv2.waitKey(1) & 0xFF == ord('q') or len(pathO) == 0:
        break



if not only_O:
    send_command_to_esp32('i')
x_flag= False
y_flag=False
centroid_buffer = []  # Global variable for storing centroid positions
correct_orientation = False

while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    cropped_frame = undistorted_frame[start_y:end_y, start_x:end_x]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

    if len(corners) > 0:
        rvec, _, _ = aruco.estimatePoseSingleMarkers(corners, 0.015, camera_matrix, dist_coeffs)
        if rvec.size != 3:
            continue
        rvec = np.array(rvec).reshape((3,))
        R, _ = cv2.Rodrigues(rvec)
        z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)
        if z_rot<0:
            z_rot=z_rot+180
        else:
            z_rot= z_rot-180
        centroid = np.mean(corners[0][0], axis=0)
        centroid_buffer.append(centroid)
        tracked.append(centroid)
        if len(centroid_buffer) > filter_size:
            centroid_buffer.pop(0)
        print(centroid_buffer[-1])
        if not only_O:
            if len(pathI) > 0:
                    x, y = pathI[0]  # Get the next target coordinates from the path

                    while not correct_orientation:
                        correct_orientation = keep_robot_fixed_orientation(z_rot)

                        # Update the current robot position and rotation
                        robot_x, robot_y = centroid_buffer[-1]
                        z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)
                        if z_rot<0:
                            z_rot=z_rot+180
                        else:
                            z_rot= z_rot-180  
                    correct_orientation = False   
                    move_robot_to_coordinates(x, y)
                            # Keep moving the robot until it reaches the desired position and orientation
                    if x_flag == True and y_flag==True: 
                        pathI.pop(0)  # Remove the visited target from the path
                        x_flag=False
                        y_flag=False
    if not only_O:
        if cv2.waitKey(1) & 0xFF == ord('q') or len(pathI) == 0:
            break
if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) != 0:
    cv2.destroyWindow('frame')

# Extracting x and y values from the second set of coordinates
x2 = [coord[0] for coord in tracked]
y2 = [coord[1] for coord in tracked]
# Plotting the coordinates
plt.plot(x1, y1, 'r', label='Given path')
plt.plot(x2, y2, 'b', label='Tracked path')

# Adding labels and titleO
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Given path vs Tracked path')

# Adding a legend
plt.legend()

# Displaying the graph
plt.show()