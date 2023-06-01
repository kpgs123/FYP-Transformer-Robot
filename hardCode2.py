import serial
import time

ser = serial.Serial('COM7', 9600, timeout=1)

def send_command_to_esp32(key, speed):
    command = key + str(speed)
    ser.write(command.encode())

    response = ser.readline().decode().rstrip()
    if response:
        print(response)

    # Wait for acknowledgment signal from ESP32
    while True:
        ack = ser.readline().decode().rstrip()
        if ack == "ack":
            break


for i in range(5):
    send_command_to_esp32('2',10)


