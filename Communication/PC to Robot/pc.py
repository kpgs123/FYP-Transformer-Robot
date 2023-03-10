import serial

port = "CNCA0"
baud_rate = 9600

ser = serial.Serial(port, baud_rate, timeout=1)

if not ser.isOpen():
    ser.open()

while True:
    data = input("Data to send: ")
    ser.write(data.encode())
    response = ser.readlines().decode().rstrip()
    print("Recieved response: ", response)