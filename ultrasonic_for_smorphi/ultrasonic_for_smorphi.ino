#include <smorphi.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

Smorphi my_robot;


#define trigFront 23
#define echoFront 25
#define trigLeft 18
#define echoLeft 19
#define trigRight 26
#define echoRight 27
#define trigBack 16
#define echoBack 17
// Threshold values
int thresholdFnt = 50;
int thresholdBk = 50;
int thresholdL = 50;
int thresholdR = 50;

String status = "o"; // declare and initialize the status variable to false


// Function declaration
int getDistance(int trigPin, int echoPin);

int getDistance(int trigPin, int echoPin) {
  int duration, distance;
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;
  return distance;
}


void setup() {  
  // put your setup code here, to run once:
  Serial.begin(115200);
  SerialBT.begin("ESP32_BT"); // Set the name for the Bluetooth device

  my_robot.BeginSmorphi();

  pinMode(trigFront, OUTPUT);  //Making pins 
  pinMode(echoFront, INPUT);
  pinMode(trigLeft, OUTPUT);
  pinMode(echoLeft, INPUT);
  pinMode(trigRight, OUTPUT);
  pinMode(echoRight, INPUT);
  pinMode(trigBack, OUTPUT);
  pinMode(echoBack, INPUT);

}

void loop() {


  String x;
  String y;
  unsigned long t0 = millis();
  

  while (millis() - t0 < 5000){
    if (SerialBT.available()) {
      x = SerialBT.read();

      if (x == "8"){
        my_robot.MoveForward(10);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved forward");
      }
      else if (x == "2"){
        my_robot.MoveBackward(10);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved backward");
      }
      else if (x == "6"){
        my_robot.MoveRight(10);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved right");
      }   
      else if (x == "4"){
        my_robot.MoveLeft(10);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved left");
      }
      else if (x == "9"){
        my_robot.MoveDiagUpRight(25);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved DiagUpRight");
      }
      else if (x == "7"){
        my_robot.MoveDiagUpLeft(25);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved DiagUpLeft");
      }
      else if (x == "3"){
        my_robot.MoveDiagDownRight(25);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved DiagDownRight");
      }
      else if (x == "1"){
        my_robot.MoveDiagDownLeft(25);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("moved DiagDownLeft");
      }   
      else if (x == "a"){
        my_robot.CenterPivotLeft(100);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("turned CenterPivotLeft");
      }
      else if (x == "c"){
        my_robot.CenterPivotRight(100);
        delay(500);
        my_robot.stopSmorphi();
        SerialBT.println("turned CenterPivotRight");
      }     
      else if (x == "o"){
        my_robot.O();
        status = "o";
        delay(100);
        SerialBT.println("Transformed to O shape");
      }
      else if (x == "l"){
        my_robot.L();
        delay(100);
        SerialBT.println("Transformed to L shape");
      }
      else if (x == "s"){
        my_robot.S();
        delay(100);
        SerialBT.println("Transformed to S shape");
      }
      else if (x == "j"){
        my_robot.J();
        delay(100);
        SerialBT.println("Transformed to J shape");
      }
      else if (x == "i"){
        my_robot.I();
        status = "i";
        delay(100);
        SerialBT.println("Transformed to I shape");
      }
      else if (x == "t"){
        my_robot.T();
        delay(100);
        SerialBT.println("Transformed to T shape");
      }
      else if (x == "z"){
        my_robot.Z();
        delay(100);
        SerialBT.println("Transformed to Z shape");
      }
      my_robot.sm_reset_M1();
      my_robot.sm_reset_M2();
      my_robot.sm_reset_M3();
      my_robot.sm_reset_M4();      
        
    }
    if (x == "o" || x== "i"){
      y = x;      
    }   
  }

   
    Serial.println("pakaya");
    
    //nt distanceF = getDistance(trigFront, echoFront);
    //Serial.println(distanceF);    
    int distanceL = getDistance(trigLeft, echoLeft);
    Serial.println(distanceL);    
    int distanceR = getDistance(trigRight, echoRight);
    Serial.println(distanceR);
    //int distanceB = getDistance(trigBack, echoBack);
    //Serial.println(distanceB);

    //delay(5);

    if (status == "i" || y == "i"){           //set the thresould values for i shape
      thresholdFnt = 20;
      thresholdBk = 40;
      thresholdL = 2;
      thresholdR = 2;

      //if (distanceF < thresholdFnt){

      if (distanceL < thresholdL){
        my_robot.stopSmorphi();
        Serial.println("Too close obstracle left");
        my_robot.MoveRight(10);
        delay(50);
        my_robot.stopSmorphi();
      }
      else if (distanceR < thresholdR){
        my_robot.stopSmorphi();
        Serial.println("Too close obstracle right");
        my_robot.MoveLeft(10);
        delay(50);
        my_robot.stopSmorphi();
      }
      //else if (distanceB < thresholdBk){

    }

    else if (status == "o" || y == "o"){           //set the thresould values for o shape
      thresholdFnt = 20;
      thresholdBk = 2;
      thresholdL = 20;
      thresholdR = 2;

      //if (distanceF < thresholdFnt){
 
      if (distanceL < thresholdL){
        my_robot.stopSmorphi();
        Serial.println("Too close obstracle left");
        my_robot.MoveRight(10);
        delay(50);
        my_robot.stopSmorphi();
      }
      else if (distanceR < thresholdR){
        my_robot.stopSmorphi();
        Serial.println("Too close obstracle right");
        my_robot.MoveLeft(10);
        delay(50);
        my_robot.stopSmorphi();
      }

  }
  
}
