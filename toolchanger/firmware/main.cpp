#include <Arduino.h>
#include <Servo.h> // servo library

void sendRoboSig(bool signal);
void changeServo(bool tool);
void changeStatus(uint8_t newStatus);
void toggleMotorPower();
void readRoboSig();
int sensor();
bool checkTool();
void checkTime();


Servo myServo; //   initialize servo lib

//  =====    variables  =====
const int lockAngle = 50;   // servo angle for locked toolchanger   (tool mounted)
const int noLockAngle = 15; // servo angle for unlocked toolchanger (no tool mounted)

const int thresh = 100;              //  threshhold for the proximity sensor
const unsigned long timeMax = 10000; //  timer for periodic check of tool status
unsigned long lastCheck = 0;         //  time of the last periodic check
int status = 0;           //  saves the status of the toolchanger (0 = no tool mounted)
bool motorActive = false;

//  =====    pin declaration    =====
const int signalLED = LED_BUILTIN; // uses the LED on the arduino to visualize status

//const int signalInPin = 2; // receives signals from the robot
const int sensorPin = A3;  // proximity sensor pin
const int servoPin = 5;    // servo pwm pin

const int relayK1 = 3; // powers on motor
//const int relayK2 = 4; // signal to robot -> emergency stop
//const int relayK3 = 7; // activates power for a mounted tool

//  =====   functions   =====

/*  send function:
*   sends a signal to the robot
*   1 = confirm
*   0 = emergency stop
*/
void sendRoboSig(bool signal)
{
    //  if input variable is true, send confirm message
    if (signal)
    {
        //digitalWrite(relayK1, HIGH);
        //delay(200);
        //digitalWrite(relayK1, LOW);
        Serial.print(status);
        Serial.println(" confirmed!");
    }

    //  else send signal to robot for an emergency stop
    else
    {
        //digitalWrite(relayK2, HIGH);
        //delay(200);
        //digitalWrite(relayK2, LOW);
        Serial.println("emergency stop");
    }
}

/*  servo switch function
*   switches the position of the servo 
*   between locked and not locked position
*   according to input
*/
void changeServo(bool tool)
{
    int angle = tool ? lockAngle : noLockAngle;
    myServo.write(angle);
    delay(500);
}

//  changes the status of the tool changer depending on the last status
void changeStatus(uint8_t newStatus)
{
    if (newStatus != status)
    {
        Serial.print("changed status from ");
        Serial.print(status);

        status = newStatus;

        changeServo(status > 0); //  switches the servo to the new position

        Serial.print(" to ");
        Serial.println(status);
    }

    //  confirms the change to the robot: true when the sensor agrees with the
    //  requested status, false (emergency stop) when it does not
    sendRoboSig(checkTool() == (status > 0));
}

void toggleMotorPower()
{
    if (motorActive)
    {
        analogWrite(relayK1, 0);
        motorActive = false;
        Serial.println("Motor Off");
    }
    else
    {
        analogWrite(relayK1, 201); //value corresponding to 4V (between required 2.5V and 5V to switch relay) 
        motorActive = true;
        Serial.println("Motor On");
    }
}

/*  read function:
*   reads the signals send from robot 
*   and starts a toolchange 
*   if a signal is received
*/
void readRoboSig()
{
    if (Serial.available() > 0)
    {
        int c = Serial.read();

        if (c < 0) //  nothing actually read
        {
            return;
        }
        else if (c == 's') //check status ('s' == status)
        {
            sendRoboSig(checkTool() == (status > 0));   //  confirms change to robot
        }
        else if (c == 'm') //toggle motor ('m' == motor)
        {
            toggleMotorPower();
        }
        else if (c >= '0' && c <= '9') //  only digits are valid status requests
        {
            changeStatus((uint8_t)(c - '0'));
        }
        //  everything else (line endings, whitespace, noise) is ignored
    }
}

/*  sensor function:
 *  reads the proximtity sensors signal and averages it over <checks> measurements
 *  to eliminate the possibility of a faulty signal
 *  then returns it
*/
int sensor()
{
    const int checks = 10; // number of checks performed
    long temp = 0;         // temporary saves the value of the measurements

    // takes multiple measurements of the sensor value
    for (int i = 0; i < checks; i++)
    {
        temp += analogRead(sensorPin); //  adds the new value to the old value
        delay(10);
    }

    temp /= checks;    //    average the value of the sensor over the number of checks performed
    return (int)temp;  //    returns the value
}

//  checks if a tool is mounted to the tool changer
bool checkTool()
{
    // a reading below the threshold means nothing is in front of the sensor
    if (sensor() < thresh)
    {
        digitalWrite(signalLED, HIGH);
        return false;
    }

    // any other value will cause the toolchanger to lock
    else
    {
        digitalWrite(signalLED, LOW);
        return true;
    }
}

/*  timer function
*   checks the time since last check 
*   and reads the state of the tool
*   sends emergency halt if not as exspected
*/
void checkTime()
{
    //  after timer runs out
    if (millis() - lastCheck >= timeMax)
    {
        //  checks if the tool is mounted like expected
        bool check = checkTool();

        if (!check && status > 0)
        {
            sendRoboSig(false); //  sends a emergency halt to the robot
        }

        lastCheck = millis(); //  resets the timer
    }
}

//  =====   setup function  =====
//  runs once at the start of the microcontroller
void setup()
{
    // start serial for readout
    Serial.begin(9600);

    // declare pin modes
    //pinMode(signalInPin, INPUT);
    pinMode(sensorPin, INPUT);
    pinMode(signalLED, OUTPUT);
    pinMode(relayK1, OUTPUT);
    //pinMode(relayK2, OUTPUT);
    //pinMode(relayK3, OUTPUT);

    //  shut pins off
    digitalWrite(signalLED, LOW);
    digitalWrite(relayK1, LOW);
    //digitalWrite(relayK2, LOW);
    //digitalWrite(relayK3, LOW);

    // connect servo pin to servo
    myServo.attach(
        servoPin, 1000,
        2000); // map the pwm signal according to the datasheet of the servo

    // start of program
    status = checkTool() ? 1 : 0; //  checks if a tool is mounted and saves this information to 'status'
    changeServo(status > 0);      //  turns the servo to the specified angle according to the state of the tool

    lastCheck = millis(); //  start the periodic check timer
}

//  =====   loop function   =====
//  repeated periodically
void loop()
{
    readRoboSig(); //  reads the signal from the robot and changes 'status' if its triggered

    checkTime(); //  checks the toolstatus periodically and sends emergency halt if needed

    delay(200); //  small delay for controller
}
