import serial
import time

ser = serial.Serial('COM8', 9600, timeout=1)

def send_command_to_esp32(command):
    ser.write(command.encode())
    response = ser.readline().decode().rstrip()
    if response:
        print(response)

def move_robot_to_coordinates(x, y):
    # Calculate the necessary movement commands to reach the given coordinates
    # Assuming the robot has a sensor or localization system to determine its position
    # Here, we assume the robot can move horizontally (x-axis) and vertically (y-axis),
    # as well as diagonally (in all four directions)

    # Move diagonally
    if x < 378 and y < 249:
        command = '7'  # Diagonal top-left
    elif x > 430 and y < 249:
        command = '9'  # Diagonal top-right
    elif x < 378 and y > 249:
        command = '1'  # Diagonal bottom-left
    elif x > 430 and y > 249:
        command = '3'  # Diagonal bottom-right
    # Move horizontally
    elif x < 378:
        command = '4'  # Left
    elif x > 430:
        command = '6'  # Right
    else:
        command = ''  # Stay in the same position horizontally

    send_command_to_esp32(command)  # Send the horizontal/diagonal movement command

    # Move vertically
    if y < 249:
        command = '2'  # Backward
    elif y > 249:
        command = '8'  # Forward
    else:
        command = ''  # Stay in the same position vertically

    send_command_to_esp32(command)  # Send the vertical movement command

path = [[378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249], [378, 249], [390, 249], [400, 249], [410, 249], [420, 249], [430, 249]]

for point in path:
    x, y = point
    move_robot_to_coordinates(x, y)
    # Wait for acknowledgment from Arduino
    while ser.readline().decode().rstrip() != 'ACK':
        pass
