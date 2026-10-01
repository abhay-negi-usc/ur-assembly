#include <Arduino.h>

#include "screwdrive.h"

//  Pins come from config/screwdrive.yaml via build_flash.sh (mtc/pins.py checks them).
#ifndef PIN_SCREWDRIVE_PWM
#error "PIN_SCREWDRIVE_PWM is not defined -- build with build_flash.sh, which takes the pins from config/screwdrive.yaml"
#endif
#ifndef PIN_SCREWDRIVE_DIR
#error "PIN_SCREWDRIVE_DIR is not defined -- build with build_flash.sh, which takes the pins from config/screwdrive.yaml"
#endif

/*  The screwdrive: a 12 V DC motor turning the coupling lead screw.
*
*   Cytron MD10C R3, sign-magnitude mode: PWM sets the speed, DIR the direction.
*
*   Wiring -- three wires, the 12 V motor supply goes straight to the MD10C's power terminals
*   and the MD10C makes its own logic supply from it:
*
*       MD10C PWM  <-  pins.pwm  (default D6: Timer0, ~980 Hz; the MD10C takes up to 20 kHz)
*       MD10C DIR  <-  pins.dir  (default D7)
*       MD10C GND  --  GND       common reference, required
*
*   The pins are set in config/screwdrive.yaml. Timer0's frequency cannot be raised without
*   breaking millis()/delay(), so on D5/D6 the motor runs at ~980 Hz and may whine a little --
*   that is harmless.
*
*   PWM LOW is BRAKE on the MD10C (both outputs pulled to ground), not coast. That is why there
*   is no back-EMF speed sensing here: the motor terminals never float, so there is no window
*   in which to measure the generated voltage.
*/

namespace screwdrive
{
namespace
{
const int pwmPin = PIN_SCREWDRIVE_PWM;
const int dirPin = PIN_SCREWDRIVE_DIR;

//  Reversing while running cuts the drive for this long before flipping DIR, so the motor is
//  never plugged straight from full forward into full reverse. (PWM LOW brakes on the MD10C.)
const unsigned long reverseDwellMs = 200;

//  How long to wait for the numbers after a command byte. The host sends the whole line in one
//  write, so it is normally all in the buffer already; a command with nothing after it times
//  out and stops.
const unsigned long argTimeoutMs = 100;

//  Longest timed run accepted, in ms (one hour). The host checks this too.
const unsigned long maxRunMs = 3600000UL;

int duty = 0; //  current signed duty, -255..255

//  what service() is timing: nothing, a timed run that stops at the end, or a ramp that holds
//  its final speed at the end
enum Mode : uint8_t
{
    IDLE,
    TIMED,
    RAMP
};
Mode mode = IDLE;
unsigned long runStart = 0;
unsigned long runMs = 0;
int rampFrom = 0;     //  ramp end points, as signed duty
int rampTo = 0;
bool runDone = false;  //  set when a timed run expires, reported from poll()
bool rampDone = false; //  set when a ramp reaches its end, reported from poll()
bool servicing = false;

void apply(int newDuty)
{
    newDuty = constrain(newDuty, -255, 255);

    bool reverse = newDuty < 0;
    if (duty != 0 && newDuty != 0 && reverse != (duty < 0))
    {
        analogWrite(pwmPin, 0);
        duty = 0;
        delay(reverseDwellMs);
    }

    digitalWrite(dirPin, reverse ? HIGH : LOW);
    analogWrite(pwmPin, abs(newDuty));
    duty = newDuty;
}

int dutyFromPercent(long pct)
{
    return (int)(pct * 255L / 100L);
}

/*  Reads exactly `count` comma-separated integers, each "<optional '-'><digits>", up to the
*   end of the line.
*
*   ALWAYS consumes through the newline, even when the numbers are bad. Anything left behind
*   would be routed as commands of its own -- and a stray digit is a coupler hold/release.
*   Returns false for a malformed line or a timeout.
*/
bool readArgs(long *out, uint8_t count)
{
    char buf[9];        //  up to 8 characters per number: "-" + 7 digits
    uint8_t n = 0;      //  characters in the current number
    uint8_t got = 0;    //  numbers completed
    bool ok = true;
    unsigned long start = millis();

    while (true)
    {
        if (millis() - start >= argTimeoutMs)
        {
            return false;
        }

        int c = Serial.read();
        if (c < 0)
        {
            continue;
        }

        bool end = c == '\n' || c == '\r';
        if (end || c == ',')
        {
            buf[n] = '\0';
            if (n == 0 || (n == 1 && buf[0] == '-') || got >= count)
            {
                ok = false;
            }
            else
            {
                out[got++] = atol(buf);
            }
            n = 0;
            if (end)
            {
                break;
            }
            continue;
        }

        bool digit = c >= '0' && c <= '9';
        bool sign = c == '-' && n == 0;
        if ((digit || sign) && n < sizeof(buf) - 1)
        {
            buf[n++] = (char)c;
        }
        else
        {
            ok = false; //  keep draining to the newline
        }
    }

    return ok && got == count;
}

//  echoes the percent as asked rather than converting duty back, which would not round-trip
void report(long pct)
{
    Serial.print("screwdrive ");
    Serial.print(pct);
    Serial.println("%");
}

//  any new command replaces a timed run or ramp in progress, without reporting it as done
void cancelRun()
{
    mode = IDLE;
    runMs = 0;
    runDone = false;
    rampDone = false;
}

/*  A ramp step. Crossing zero goes through a stop first, so apply() sees a start from rest and
*   does not insert its 200 ms reversing brake into the middle of the ramp -- near zero the
*   motor is barely turning anyway.
*/
void rampStep(int newDuty)
{
    if (newDuty == duty)
    {
        return;
    }
    if (duty != 0 && newDuty != 0 && (newDuty < 0) != (duty < 0))
    {
        apply(0);
    }
    apply(newDuty);
}

/*  Runs the motor at a signed duty for `ms`, then stops it. Returns at once; service() ends
*   the run, so the coupler keeps working while the motor turns.
*/
void runFor(int newDuty, unsigned long ms)
{
    cancelRun();
    apply(newDuty);
    runStart = millis(); //  after any reverse dwell, so the run gets its full time
    runMs = ms;
    mode = TIMED;
}

//  Ramps linearly from one duty to another over `ms`, then holds the second. Returns at once.
void rampFor(int from, int to, unsigned long ms)
{
    cancelRun();
    apply(from);
    rampFrom = from;
    rampTo = to;
    runStart = millis();
    runMs = ms;
    mode = RAMP;
}

//  reads "<value>,...,<ms>" -- `count` numbers, the last a duration -- or stops and says why
bool readRun(long *args, uint8_t count)
{
    if (!readArgs(args, count) || args[count - 1] <= 0 ||
        (unsigned long)args[count - 1] > maxRunMs)
    {
        Serial.println("screwdrive rejected a malformed run, stopping");
        cancelRun();
        apply(0);
        report(0);
        return false;
    }
    return true;
}
}

void setup()
{
    //  Driven low before anything else. Through reset and the bootloader these pins float, so
    //  fit a 10k pull-down from PWM to GND to keep the MD10C off in that window.
    pinMode(pwmPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    digitalWrite(pwmPin, LOW);
    digitalWrite(dirPin, LOW);
    duty = 0;
    cancelRun();
}

bool handle(int c)
{
    long args[3];

    if (c == 'd') //set screwdrive speed ('d' == drive)
    {
        cancelRun();
        if (!readArgs(args, 1))
        {
            //  fail stopped: a garbled speed must never leave the motor running at the old one
            Serial.println("screwdrive rejected a malformed speed, stopping");
            args[0] = 0;
        }
        long pct = constrain(args[0], -100L, 100L);
        apply(dutyFromPercent(pct));
        report(pct);
    }
    else if (c == 't') //timed run at a percent ('t' == timed): "t<pct>,<ms>"
    {
        if (readRun(args, 2))
        {
            long pct = constrain(args[0], -100L, 100L);
            runFor(dutyFromPercent(pct), args[1]);
            Serial.print("screwdrive ");
            Serial.print(pct);
            Serial.print("% for ");
            Serial.print(runMs);
            Serial.println(" ms");
        }
    }
    else if (c == 'p') //timed run at a raw duty ('p' == pwm): "p<duty>,<ms>", duty -255..255
    {
        //  The host's open-loop rpm command uses this: it converts rpm to duty from the max rpm
        //  in its config, so the motor's speed rating lives in one place and is not baked in here.
        if (readRun(args, 2))
        {
            runFor((int)constrain(args[0], -255L, 255L), args[1]);
            Serial.print("screwdrive pwm ");
            Serial.print(duty);
            Serial.print(" for ");
            Serial.print(runMs);
            Serial.println(" ms");
        }
    }
    else if (c == 'a') //ramp ('a' == accelerate): "a<pct1>,<pct2>,<ms>", then hold pct2
    {
        if (readRun(args, 3))
        {
            long from = constrain(args[0], -100L, 100L);
            long to = constrain(args[1], -100L, 100L);
            rampFor(dutyFromPercent(from), dutyFromPercent(to), args[2]);
            Serial.print("screwdrive ramp ");
            Serial.print(from);
            Serial.print("% to ");
            Serial.print(to);
            Serial.print("% over ");
            Serial.print(runMs);
            Serial.println(" ms");
        }
    }
    else
    {
        return false;
    }
    return true;
}

void service()
{
    //  delay() inside apply() calls yield(), which calls this again
    if (servicing)
    {
        return;
    }
    servicing = true;

    unsigned long elapsed = millis() - runStart;
    if (mode == TIMED && elapsed >= runMs)
    {
        cancelRun();
        apply(0); //  going to zero never dwells, so this cannot block
        runDone = true;
    }
    else if (mode == RAMP && elapsed >= runMs)
    {
        cancelRun();
        rampStep(rampTo); //  and stays there, like drive
        rampDone = true;
    }
    else if (mode == RAMP)
    {
        //  at most 510 * 3600000 = 1.8e9, inside a signed long
        rampStep(rampFrom + (int)((long)(rampTo - rampFrom) * (long)elapsed / (long)runMs));
    }

    servicing = false;
}

void poll()
{
    service();

    if (runDone)
    {
        runDone = false;
        Serial.println("screwdrive run done");
    }
    if (rampDone)
    {
        rampDone = false;
        Serial.println("screwdrive ramp done");
    }
}
}
