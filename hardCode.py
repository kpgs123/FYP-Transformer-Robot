import serial
import time

ser = serial.Serial('COM8', 9600, timeout=1)

def send_command_to_esp32(command):
    # Wait for a brief moment to ensure that the serial port is ready
    #time.sleep(1)

    # Send a command to the ESP32
    ser.write(command.encode())

    # Wait for a response from the ESP32
    response = ser.readline().decode().rstrip()
    if response:
        print(response)

# Example usage
i=0
while i<=10:
    i+=1
    command = '4'  
    send_command_to_esp32(command)
