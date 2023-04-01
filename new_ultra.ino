// Define the pins for the ultrasonic sensors
const int trigPinFront = 9;
const int echoPinFront = 10;
const int trigPinLeft = 11;
const int echoPinLeft = 12;
const int trigPinRight = 13;
const int echoPinRight = A0;
const int trigPinBack = A1;
const int echoPinBack = A2;

// Define variables to store the distances
int distanceFront;
int distanceLeft;
int distanceRight;
int distanceBack;

void setup() {
  // Set the trigger pins as outputs and the echo pins as inputs
  pinMode(trigPinFront, OUTPUT);
  pinMode(echoPinFront, INPUT);
  pinMode(trigPinLeft, OUTPUT);
  pinMode(echoPinLeft, INPUT);
  pinMode(trigPinRight, OUTPUT);
  pinMode(echoPinRight, INPUT);
  pinMode(trigPinBack, OUTPUT);
  pinMode(echoPinBack, INPUT);
  
  // Begin serial communication at 9600 baud
  Serial.begin(9600);
}

void loop() {
  // Read the distances from each ultrasonic sensor
  distanceFront = readDistance(trigPinFront, echoPinFront);
  distanceLeft = readDistance(trigPinLeft, echoPinLeft);
  distanceRight = readDistance(trigPinRight, echoPinRight);
  distanceBack = readDistance(trigPinBack, echoPinBack);
  
  // Print the distances to the serial monitor
  Serial.print("Front Distance: ");
  Serial.print(distanceFront);
  Serial.print("\tLeft Distance: ");
  Serial.print(distanceLeft);
  Serial.print("\tRight Distance: ");
  Serial.print(distanceRight);
  Serial.print("\tBack Distance: ");
  Serial.println(distanceBack);
}

int readDistance(int trigPin, int echoPin) {
  // Send a 10 microsecond pulse to the trigger pin to start the measurement
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Measure the time delay between sending the pulse and receiving the echo
  int duration = pulseIn(echoPin, HIGH);
  
  // Calculate the distance based on the time delay and the speed of sound
  int distance = duration * 0.034 / 2;
  
  // Return the distance value
  return distance;
}
