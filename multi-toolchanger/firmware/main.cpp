#include <Arduino.h>

#include "coupler.h"
#include "screwdrive.h"
#include "relay.h"

/*  One Uno, several devices. Each device is a module with the same shape:
*
*       setup()        pins and initial state
*       handle(c)      true if the serial byte `c` was one of its commands
*       poll()         periodic work, for the modules that need it
*
*   This file only boots the modules and routes bytes to them. Adding a device is a new
*   module, a setup() call below and one more `||` in loop(). Command bytes must not clash:
*
*       coupler   '0'..'9'  's'  'r'  'b'
*       relay     'm'
*       screwdrive  'd<pct>\n'  't<pct>,<ms>\n'  'p<duty>,<ms>\n'  'a<p1>,<p2>,<ms>\n'
*/

//  =====   setup function  =====
//  runs once at the start of the microcontroller
void setup()
{
    Serial.begin(9600);

    //  anything that switches power goes to its safe (off) state first, before the slow work
    screwdrive::setup();
    relay::setup();

    //  Announced BEFORE the coupler's slow work, not after. The host uses this line to know the
    //  board has just reset, and checkTool() (100 ms) plus changeServo() (500 ms) on top of
    //  the bootloader pushed it far enough out that the host gave up waiting for it.
    Serial.println("toolchanger ready");

    coupler::setup();
}

/*  The Arduino core's delay() calls yield() on every pass of its wait, so this keeps the motor's
*   timed runs ending on time through the coupler's blocking waits -- a servo move is 500 ms and
*   waitForTool() can take 1.5 s. Without it a timed run could overrun by seconds.
*
*   Only the motor's service() runs here: it never prints, so it cannot land in the middle of a
*   line the coupler is halfway through writing.
*/
void yield()
{
    screwdrive::service();
}

//  =====   loop function   =====
//  repeated periodically
void loop()
{
    if (Serial.available() > 0)
    {
        int c = Serial.read();

        //  everything no module claims (line endings, whitespace, noise) is ignored
        if (c >= 0)
        {
            coupler::handle(c) || relay::handle(c) || screwdrive::handle(c);
        }
    }

    coupler::poll(); //  checks the toolstatus periodically and sends emergency halt if needed
    screwdrive::poll(); //  ends timed runs and reports them

    delay(200); //  small delay for controller
}
