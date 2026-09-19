#include <Arduino.h>
#include <Servo.h> // servo library

void sendRoboSig(bool signal);
void changeServo(bool tool);
void changeStatus(uint8_t newStatus);
void toggleMotorPower();
void readRoboSig();
int sensor();
bool toolFrom(int value);
bool checkTool();
bool waitForTool(bool want);
void reportSensor();
void checkTime();


Servo myServo; //   initialize servo lib

//  =====    variables  =====
const int lockAngle = 50;   // servo angle for locked toolchanger   (tool mounted)
const int noLockAngle = 15; // servo angle for unlocked toolchanger (no tool mounted)

/*  proximity sensor calibration
*   The Uno's ADC is 10 bit, so analogRead() returns 0..1023. An earlier version of this
*   sketch said the no-tool reading would be 4095 -- that is a 12 bit value from a different
*   board, so the threshold below was never tuned for THIS hardware, and the comment
*   disagreed with the code about which side means "tool". That is what makes the sensor
*   "disagree with the commanded state".
*
*   To calibrate, with no risk of the servo moving:
*       ./toolchanger.py calibrate
*   or send 'r' by hand with a tool mounted and again with nothing mounted. Put the midpoint
*   of the two readings in thresh, and set toolReadsHigh to match which reading is larger.
*/
/*  MEASURED on this cell: with the changer EMPTY the pin sits railed at 1022..1023, stable
*   to within a count. The rail is therefore the NO-TOOL reading -- which is what the original
*   sketch's "the value will be 4095" comment meant, 4095 being the rail on the 12 bit board it
*   was ported from. A tool in front of the probe pulls the reading DOWN, so toolReadsHigh is
*   false here; it was true, which is why the board reported a tool while gripping nothing.
*
*   thresh is still PROVISIONAL: 1023 is the only state measured so far. Mount a tool and run
*   `./toolchanger.py calibrate` to replace it with the real midpoint.
*/
const int thresh = 900;           //  below this = tool present, above = empty (see above)
const bool toolReadsHigh = false; //  true:  a mounted tool reads ABOVE thresh
                                  //  false: a mounted tool reads BELOW thresh (this probe)

/*  How long to let the sensor come around after a move before calling it an emergency stop.
*   The tool needs a moment to seat once the servo has swung, and a single unlucky sample
*   used to be reported as a failure -- which is why the disagreement looked intermittent.
*/
const unsigned long settleMax = 1500;

const unsigned long timeMax = 10000; //  timer for periodic check of tool status
unsigned long lastCheck = 0;         //  time of the last periodic check
int status = 0;           //  saves the status of the toolchanger (0 = no tool mounted)
int lastRaw = 0;          //  most recent averaged reading, reported on an emergency stop
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
        //  report the reading behind it, so the host can tell a miscalibrated threshold
        //  apart from a tool that genuinely is not there
        Serial.print("emergency stop raw=");
        Serial.print(lastRaw);
        Serial.print(" thresh=");
        Serial.println(thresh);
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
        //  printed in one go, before the servo blocks -- splitting it around changeServo()
        //  left half a line on the wire for 500 ms and the host read it as two lines
        Serial.print("changed status from ");
        Serial.print(status);
        Serial.print(" to ");
        Serial.println(newStatus);

        status = newStatus;

        changeServo(status > 0); //  switches the servo to the new position
    }

    /*  Locking and releasing are not mirror images of each other.
    *
    *   Locking: the bearings must have something to grip. If the sensor sees nothing we
    *   have clamped thin air, and that IS an emergency.
    *
    *   Releasing: the bearings retract, but the tool goes on sitting in the changer until
    *   something physically pulls it away. The sensor still seeing it is the NORMAL case,
    *   so demanding an empty reading here turned every good release into an emergency stop.
    *   Report what the sensor sees, and confirm the release.
    */
    if (status > 0)
    {
        sendRoboSig(waitForTool(true));
    }
    else
    {
        Serial.print("released, tool ");
        Serial.println(checkTool() ? "still in the changer" : "gone");
        sendRoboSig(true);
    }
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
            //  a query reports what the sensor says right now, with no retry -- unlike a
            //  commanded change, nothing is expected to be settling
            bool present = checkTool();

            Serial.print("tool ");
            Serial.println(present ? "present" : "absent");

            //  only a changer CLAIMING to hold a tool that is not there is an emergency;
            //  a released changer with the tool still resting in it is perfectly normal
            sendRoboSig(status == 0 || present);
        }
        else if (c == 'r') //raw sensor value ('r' == raw), for calibration
        {
            reportSensor();
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

    temp /= checks;       //    average the value of the sensor over the number of checks performed
    lastRaw = (int)temp;  //    kept for the emergency stop message
    return lastRaw;       //    returns the value
}

//  which side of the threshold counts as "tool present" depends on the probe wiring
bool toolFrom(int value)
{
    return toolReadsHigh ? (value >= thresh) : (value < thresh);
}

//  checks if a tool is mounted to the tool changer
bool checkTool()
{
    bool present = toolFrom(sensor());

    //  the LED lights when nothing is mounted
    digitalWrite(signalLED, present ? LOW : HIGH);
    return present;
}

/*  waits for the sensor to agree with `want`, up to settleMax
*   Each checkTool() averages 10 readings 10 ms apart, so this retries about every 100 ms.
*   Returns true as soon as the sensor agrees, false if it never does.
*/
bool waitForTool(bool want)
{
    unsigned long start = millis();

    do
    {
        if (checkTool() == want)
        {
            return true;
        }
    } while (millis() - start < settleMax);

    return false;
}

//  prints the raw averaged reading, for calibrating thresh and toolReadsHigh
void reportSensor()
{
    int value = sensor();

    Serial.print("raw ");
    Serial.print(value);
    Serial.print(" thresh ");
    Serial.print(thresh);
    Serial.print(" tool ");
    Serial.print(toolFrom(value) ? "yes" : "no");
    Serial.print(" status ");
    Serial.println(status);
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
        //  checks if the tool is mounted like expected (also refreshes the LED)
        bool check = checkTool();

        //  only alarm when the tool is still missing after a retry, so a single noisy
        //  sample does not fire a spurious halt while idling
        if (!check && status > 0 && !waitForTool(true))
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

    //  Announced BEFORE the slow work below, not after. The host uses this line to know the
    //  board has just reset, and checkTool() (100 ms) plus changeServo() (500 ms) on top of
    //  the bootloader pushed it far enough out that the host gave up waiting for it.
    Serial.println("toolchanger ready");

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
