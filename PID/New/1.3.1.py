def move_robot_to_coordinates(x, y):
    global centroid_buffer  # Declare centroid_buffer as a global variable

    # Get the current robot position
    robot_x, robot_y = centroid_buffer[-1]  # Assuming centroid_buffer contains the robot's position

    # Calculate the necessary movement commands to reach the given coordinates
    dx = x - robot_x
    dy = y - robot_y

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
