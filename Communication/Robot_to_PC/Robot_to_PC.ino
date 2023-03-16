#include <smorphi.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

Smorphi my_robot;

void setup() {  
  // put your setup code here, to run once:
  Serial.begin(115200);
  SerialBT.begin("ESP32_BT"); // Set the name for the Bluetooth device

  my_robot.BeginSmorphi();


  /*Serial.println("moving forward");
  my_robot.MoveForward(10); // robot moves forward with speed 10 for 2 seconds
  delay(2000);
  my_robot.stopSmorphi();
  delay(500);
  Serial.println("transforming to l shape");
  my_robot.L();
  delay(5000);
  my_robot.MoveBackward(10);
  delay(2000);
  my_robot.stopSmorphi();
  my_robot.O();*/


}

void loop() {
  char x;

  if (SerialBT.available()) {
    x = SerialBT.read();
    if (x == 'O'){
      my_robot.stopSmorphi();
      delay(500);
      Serial.println("transforming to o shape");
      my_robot.O();
      delay(5000);
    }
    else if (x == 'L'){
      my_robot.stopSmorphi();
      delay(500);
      Serial.println("transforming to o shape");
      my_robot.L();
      delay(5000);
    }
    else if (x == 'S'){
      my_robot.stopSmorphi();
      delay(500);
      Serial.println("transforming to o shape");
      my_robot.S();
      delay(5000);
    }
    else if (x == 'J'){
      my_robot.stopSmorphi();
      delay(500);
      Serial.println("transforming to o shape");
      my_robot.J();
      delay(5000);
    }
    else if (x == 'I'){
      my_robot.stopSmorphi();
      delay(500);
      Serial.println("transforming to o shape");
      my_robot.I();
      delay(5000);
    }
     SerialBT.write('R');
     delay(100);
  }
  
}
