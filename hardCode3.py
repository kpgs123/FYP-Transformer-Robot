import serial
import time



previous_command_time = 0  # Variable to store the time of the previous command

def send_command_to_esp32(key, speed):
    global previous_command_time

    # Calculate the time difference between the current command and the previous command
    current_time = time.time()
    time_diff = current_time - previous_command_time

    if time_diff < .5:
        # If the time difference is less than 500ms, wait for the remaining time
        print("before sleep" + str(time.time()))
        time.sleep(.5 - time_diff)
        print(time.time())


    # Combine key and speed into a command string
    command = key + str(speed)

    # Send the command to the ESP32
    print(command)



    previous_command_time = time.time()  # Update the previous command time

for i in range(5):
    send_command_to_esp32('5',20)




