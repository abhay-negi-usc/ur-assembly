#include <Arduino.h>
#include <Servo.h> // servo library

#include "coupler.h"

namespace coupler
{
namespace
{
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
*       ./multitoolchanger.py calibrate
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
*   `./multitoolchanger.py calibrate` to replace it with the real midpoint.
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

/*  Sensor bypass: when true, hold and release move the servo and confirm without asking the
*   proximity sensor, and the periodic watchdog stops alarming.
*
*   This exists for working on the mechanism while the sensor is untrustworthy -- a probe that
*   is unplugged or shorted reads a hard 0 or a hard 1023 and then "agrees" with everything,
*   which is worse than useless because it confirms grips that are not happening.
*
*   It gives up the emergency stop, so it is deliberately NOT persistent: it lives in RAM and
*   every reset -- including every time the host opens the serial port -- clears it back to
*   false. You cannot leave the machine bypassed by accident.
*/
bool sensorBypass = false;

//  =====    pin declaration    =====
const int signalLED = LED_BUILTIN; // uses the LED on the arduino to visualize status
const int sensorPin = A3;          // proximity sensor pin
const int servoPin = 5;            // servo pwm pin

//  =====   functions   =====

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
        Serial.print(status);
        Serial.println(" confirmed!");
    }

    //  else send signal to robot for an emergency stop
    else
    {
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
    if (sensorBypass)
    {
        //  no sensor opinion is sought; the servo was commanded and that is all we claim
        Serial.println("(sensor bypassed)");
        sendRoboSig(true);
    }
    else if (status > 0)
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

//  turns the sensor check off or back on; cleared by any reset
void toggleBypass()
{
    sensorBypass = !sensorBypass;

    Serial.print("sensor bypass ");
    Serial.println(sensorBypass ? "ON -- grip is NOT verified" : "off");
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
    Serial.print(status);
    Serial.print(" bypass ");
    Serial.println(sensorBypass ? "on" : "off");
}
}

void setup()
{
    pinMode(sensorPin, INPUT);
    pinMode(signalLED, OUTPUT);
    digitalWrite(signalLED, LOW);

    // connect servo pin to servo
    myServo.attach(
        servoPin, 1000,
        2000); // map the pwm signal according to the datasheet of the servo

    // start of program
    status = checkTool() ? 1 : 0; //  checks if a tool is mounted and saves this information to 'status'
    changeServo(status > 0);      //  turns the servo to the specified angle according to the state of the tool

    lastCheck = millis(); //  start the periodic check timer
}

/*  reads the signals send from robot
*   and starts a toolchange
*   if a signal is received
*/
bool handle(int c)
{
    if (c == 's') //check status ('s' == status)
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
    else if (c == 'b') //toggle sensor bypass ('b' == bypass)
    {
        toggleBypass();
    }
    else if (c >= '0' && c <= '9') //  only digits are valid status requests
    {
        changeStatus((uint8_t)(c - '0'));
    }
    else
    {
        return false;
    }
    return true;
}

/*  timer function
*   checks the time since last check
*   and reads the state of the tool
*   sends emergency halt if not as exspected
*/
void poll()
{
    //  after timer runs out
    if (millis() - lastCheck >= timeMax)
    {
        //  checks if the tool is mounted like expected (also refreshes the LED)
        bool check = checkTool();

        //  only alarm when the tool is still missing after a retry, so a single noisy
        //  sample does not fire a spurious halt while idling
        if (!sensorBypass && !check && status > 0 && !waitForTool(true))
        {
            sendRoboSig(false); //  sends a emergency halt to the robot
        }

        lastCheck = millis(); //  resets the timer
    }
}
}
