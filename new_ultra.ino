// Ultrasonic Sensor pins
#define trigPin1 2
#define echoPin1 4
#define trigPin2 15
#define echoPin2 13
#define trigPin3 16
#define echoPin3 17
#define trigPin4 18
#define echoPin4 19

// Threshold values
#define threshold1 30
#define threshold2 30
#define threshold3 30
#define threshold4 30

void setup() {
  Serial.begin(9600);
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);
  pinMode(trigPin3, OUTPUT);
  pinMode(echoPin3, INPUT);
  pinMode(trigPin4, OUTPUT);
  pinMode(echoPin4, INPUT);
}

void loop() {
  long duration1, distance1;
  digitalWrite(trigPin1, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin1, LOW);
  duration1 = pulseIn(echoPin1, HIGH);
  distance1 = duration1 * 0.034 / 2;

  long duration2, distance2;
  digitalWrite(trigPin2, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin2, LOW);
  duration2 = pulseIn(echoPin2, HIGH);
  distance2 = duration2 * 0.034 / 2;

  long duration3, distance3;
  digitalWrite(trigPin3, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin3, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin3, LOW);
  duration3 = pulseIn(echoPin3, HIGH);
  distance3 = duration3 * 0.034 / 2;

  long duration4, distance4;
  digitalWrite(trigPin4, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin4, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin4, LOW);
  duration4 = pulseIn(echoPin4, HIGH);
  distance4 = duration4 * 0.034 / 2;

  // Check for obstacles
  if (distance1 < threshold1) {
    Serial.println("Obstacle detected in front");
  }
  if (distance2 < threshold2) {
    Serial.println("Obstacle detected on the left");
  }
  if (distance3 < threshold3) {
    Serial.println("Obstacle detected on the right");
  }
  if (distance4 < threshold4) {
    Serial.println("Obstacle detected behind");
  }

  delay(500);
}
