import serial
import time

# Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
ser = serial.Serial('COM4', 9600, timeout=1)

# Send data from Python to ESP32
data = "Hello, ESP32!"
ser.write(data.encode())

while True:
# Receive data from ESP32 in Python
    data = ser.readline()
    s = data.decode()
    if s != "":
        print(s, end="")
        if s[:2] == "Hi":
            data = "What is your name?"
            ser.write(data.encode())

    time.sleep(1)