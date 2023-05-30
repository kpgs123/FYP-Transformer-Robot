#include <smorphi.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

Smorphi my_robot;

String status = "p"; // declare and initialize the status variable to false

void setup() {  
  Serial.begin(115200);
  SerialBT.begin("ESP32_BT"); // Set the name for the Bluetooth device

  my_robot.BeginSmorphi();
}

void loop() {
  char x;

  if (SerialBT.available()) {
    x = SerialBT.read();

    if (x == '8'){
      my_robot.MoveForward(10);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved forward");
    }
    else if (x == '2'){
      my_robot.MoveBackward(10);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved backward");
    }
    else if (x == '6'){
      my_robot.MoveRight(10);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved right");
    }   
    else if (x == '4'){
      my_robot.MoveLeft(10);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved left");
    }
    else if (x == '9'){
      my_robot.MoveDiagUpRight(25);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved DiagUpRight");
    }
    else if (x == '7'){
      my_robot.MoveDiagUpLeft(25);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved DiagUpLeft");
    }
    else if (x == '3'){
      my_robot.MoveDiagDownRight(25);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved DiagDownRight");
    }
    else if (x == '1'){
      my_robot.MoveDiagDownLeft(25);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("moved DiagDownLeft");
    }   
    else if (x == 'a'){
      my_robot.CenterPivotLeft(50);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("turned CenterPivotLeft");
    }
    else if (x == 'c'){
      my_robot.CenterPivotRight(50);
      delay(500);
      my_robot.stopSmorphi();
      SerialBT.println("turned CenterPivotRight");
    }     
    else if (x == 'o'){
      my_robot.O();
      status = "o";
      delay(100);
      SerialBT.println("Transformed to O shape");
    }
    else if (x == 'l'){
      my_robot.L();
      delay(100);
      SerialBT.println("Transformed to L shape");
    }
    else if (x == 's'){
      my_robot.S();
      delay(100);
      SerialBT.println("Transformed to S shape");
    }
    else if (x == 'j'){
      my_robot.J();
      delay(100);
      SerialBT.println("Transformed to J shape");
    }
    else if (x == 'i'){
      my_robot.I();
      status = "i";
      delay(100);
      SerialBT.println("Transformed to I shape");
    }
    else if (x == 't'){
      my_robot.T();
      delay(100);
      SerialBT.println("Transformed to T shape");
    }
    else if (x == 'z'){
      my_robot.Z();
      delay(100);
      SerialBT.println("Transformed to Z shape");
    }
    //my_robot.sm_reset_M1();
    //my_robot.sm_reset_M2();
    //my_robot.sm_reset_M3();
    //my_robot.sm_reset_M4();
  }
}
