correct_orientation = False


def keep_robot_fixed_orientation(z_rot):
    target_angle = 0  # Desired target angle

    # Check if the current rotation angle is within the desired range
    if z_rot > 10:
        command = 'c'  # Rotate clockwise
    elif z_rot > -170:
        command = 'a'  # Rotate anticlockwise
    else:
        command = ''  # Stay in the same orientation

    send_command_to_esp32(command)  # Send the rotation command to the robot

    # Return True if the robot is in the correct orientation, False otherwise
    return z_rot <= 10 or z_rot <=-170 



if len(path) > 0:
        x, y = path[0]  # Get the next target coordinates from the path

        # Keep moving the robot until it reaches the desired position and orientation
        while not correct_orientation:
            correct_orientation = keep_robot_fixed_orientation(z_rot)
           # Update the current robot position and rotation
            z_rot = round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)

        path.pop(0)