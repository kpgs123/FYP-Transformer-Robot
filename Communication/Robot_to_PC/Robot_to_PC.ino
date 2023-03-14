#include <HardwareSerial.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(9600);
  SerialBT.begin("ESP32 Bluetooth Serial Port"); // set the Bluetooth name
  Serial.println("Bluetooth Serial Port is ready.");
}

void loop() {
  if (Serial.available()) {
    SerialBT.write(Serial.read()); // send serial data to Bluetooth
  }
  if (SerialBT.available()) {
    Serial.write(SerialBT.read()); // send Bluetooth data to serial
  }
}
