import serial
import time

# Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
ser = serial.Serial('COM9', 9600, timeout=1)

while True:
# Receive data from ESP32 in Python
    s = input()
    data = s
    ser.write(data.encode())
    time.sleep(1)
    data = ser.readline()
    s = data.decode()
    print(s[:2])
