import keyboard
import serial
import time

# Replace "/dev/tty.SLAB_USBtoUART" with the Bluetooth serial port of your ESP32
ser = serial.Serial('COM7', 9600, timeout=1)

# Define a variable to keep track of the last key press time
last_key_press_time = time.monotonic()

# Define a callback function to handle key presses8
def on_key_press(event):
    global last_key_press_time
    elapsed_time = time.monotonic() - last_key_press_time
    if elapsed_time >= 0.5:
        pressed_key = event.name
        ser.write(pressed_key.encode())
        last_key_press_time = time.monotonic()

# Set up a listener for key presses
keyboard.on_press(on_key_press)

# Keep the program running to allow key presses to be detected
while True:
    data = ser.readline()
    s = data.decode()
    s = s[:-2]
    if len(s):
        print(s)
