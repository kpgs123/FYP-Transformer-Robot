import serial
import time

ser = serial.Serial('COM8', 9600, timeout=1)

def send_command_to_esp32(command):
    ser.write(command.encode())
    response = ser.readline().decode().rstrip()
    if response:
        print(response)

i = 0
while i <= 10:
    i += 1
    command = '4'
    send_command_to_esp32(command)
    # Wait for acknowledgment from Arduino
    while ser.readline().decode().rstrip() != 'ACK':
        pass
    
