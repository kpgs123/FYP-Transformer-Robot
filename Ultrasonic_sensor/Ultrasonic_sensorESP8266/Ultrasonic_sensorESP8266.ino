// Ultrasonic Sensor pins
#define trigFront 5
#define echoFront 16
#define trigLeft 4
#define echoLeft 2
#define trigRight 14
#define echoRight 12
#define trigBack 13
#define echoBack 15

// Threshold values
#define thresholdFnt 30
#define thresholdBk 30
#define thresholdL 30
#define thresholdR 30

void setup() {
  Serial.begin(115200);
  pinMode(trigFront, OUTPUT);
  pinMode(echoFront, INPUT);
  pinMode(trigLeft, OUTPUT);
  pinMode(echoLeft, INPUT);
  pinMode(trigRight, OUTPUT);
  pinMode(echoRight, INPUT);
  pinMode(trigBack, OUTPUT);
  pinMode(echoBack, INPUT);
}

void loop() {
  long duration1, distance1;
  digitalWrite(trigFront, LOW);
  delayMicroseconds(2);
  digitalWrite(trigFront, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigFront, LOW);
  duration1 = pulseIn(echoFront, HIGH);
  distance1 = duration1 * 0.034 / 2;

  long duration2, distance2;
  digitalWrite(trigLeft, LOW);
  delayMicroseconds(2);
  digitalWrite(trigLeft, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigLeft, LOW);
  duration2 = pulseIn(echoLeft, HIGH);
  distance2 = duration2 * 0.034 / 2;

  long duration3, distance3;
  digitalWrite(trigRight, LOW);
  delayMicroseconds(2);
  digitalWrite(trigRight, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigRight, LOW);
  duration3 = pulseIn(echoRight, HIGH);
  distance3 = duration3 * 0.034 / 2;

  long duration4, distance4;
  digitalWrite(trigBack, LOW);
  delayMicroseconds(2);
  digitalWrite(trigBack, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigBack, LOW);
  duration4 = pulseIn(echoBack, HIGH);
  distance4 = duration4 * 0.034 / 2;

  Serial.print("Distance 1: ");
  Serial.print(distance1);
  Serial.println(" cm");

  Serial.print("Distance 2: ");
  Serial.print(distance2);
  Serial.println(" cm");  

  Serial.print("Distance 3: ");
  Serial.print(distance3);
  Serial.println(" cm");

  Serial.print("Distance 4: ");
  Serial.print(distance4);
  Serial.println(" cm");  


  // Check for obstacles


  delay(5);
}
