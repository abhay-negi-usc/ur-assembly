#include <Arduino.h>

/*  One Uno, several devices. Each device is a module with the same shape:
*
*       setup()        pins and initial state
*       handle(c)      true if the serial byte `c` was one of its commands
*       poll()         periodic work, for the modules that need it
*
*   WHICH MODULES ARE BUILT IN is chosen at compile time: build_flash.sh defines MODULE_<NAME>
*   for each module in the config's `modules:` list (or its --modules), and compiles only those
*   .cpp files. The board then announces what it has, so the host can load the same modules.
*
*   This file only boots the modules and routes bytes to them. Adding a device is a new module,
*   an #ifdef'd block in each function below and its name in printIdentity(). Command bytes
*   must not clash:
*
*       main        '?'
*       coupler     '0'..'9'  's'  'r'  'b'
*       relay       'm'
*       screwdrive  'd<pct>\n'  't<pct>,<ms>\n'  'p<duty>,<ms>\n'  'a<p1>,<p2>,<ms>\n'
*       t74         'J' 'L' 'R' 'A' 'H' 'Z' 'S' 'X' 'O' 'I' 'E' (upper case, see t74.h)
*
*   PINS are not chosen here: each module's come from its config yaml as PIN_<MODULE>_<ROLE>.
*   build_flash.sh checks them first (mtc/pins.py) -- capabilities, clashes, and timers a
*   module takes over (the coupler's Servo and the t74's control loop both need Timer1) -- and
*   refuses to build a combination that cannot work.
*/

#if !defined(MODULE_COUPLER) && !defined(MODULE_RELAY) && !defined(MODULE_SCREWDRIVE) && \
    !defined(MODULE_T74)
#error "no modules selected -- build with build_flash.sh, which defines MODULE_<NAME>"
#endif

#ifdef MODULE_COUPLER
#include "coupler.h"
#endif
#ifdef MODULE_RELAY
#include "relay.h"
#endif
#ifdef MODULE_SCREWDRIVE
#include "screwdrive.h"
#endif
#ifdef MODULE_T74
#include "t74.h"
#endif

//  The wire protocol's version. BUMP IT whenever any command or reply changes shape, and the
//  host's PROTOCOL in mtc/config.py with it: the host refuses a board on a different version
//  rather than sending it commands it would misread.
const int PROTOCOL = 5;

//  "modules=coupler,screwdrive proto=2" -- what this build has. The host parses it from the
//  boot banner, or asks again with '?'.
void printIdentity()
{
    bool first = true;
    Serial.print("modules=");
#ifdef MODULE_COUPLER
    Serial.print(first ? "" : ",");
    Serial.print("coupler");
    first = false;
#endif
#ifdef MODULE_RELAY
    Serial.print(first ? "" : ",");
    Serial.print("relay");
    first = false;
#endif
#ifdef MODULE_SCREWDRIVE
    Serial.print(first ? "" : ",");
    Serial.print("screwdrive");
    first = false;
#endif
#ifdef MODULE_T74
    Serial.print(first ? "" : ",");
    Serial.print("t74");
    first = false;
#endif
    (void)first;
    Serial.print(" proto=");
    Serial.println(PROTOCOL);
}

//  =====   setup function  =====
//  runs once at the start of the microcontroller
void setup()
{
    Serial.begin(115200);

    //  anything that switches power goes to its safe (off) state first, before the slow work
#ifdef MODULE_SCREWDRIVE
    screwdrive::setup();
#endif
#ifdef MODULE_RELAY
    relay::setup();
#endif
#ifdef MODULE_T74
    t74::setup();
#endif

    //  Announced BEFORE the coupler's slow work, not after. The host uses this line to know the
    //  board has just reset, and checkTool() (100 ms) plus changeServo() (500 ms) on top of
    //  the bootloader pushed it far enough out that the host gave up waiting for it.
    Serial.print("toolchanger ready ");
    printIdentity();

#ifdef MODULE_COUPLER
    coupler::setup();
#endif
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
#ifdef MODULE_SCREWDRIVE
    screwdrive::service();
#endif
}

//  =====   loop function   =====
//  repeated periodically
void loop()
{
    if (Serial.available() > 0)
    {
        int c = Serial.read();

        //  everything no module claims (line endings, whitespace, noise) is ignored
        if (c == '?') //identity ('?' == what are you)
        {
            printIdentity();
        }
        else if (c >= 0)
        {
            bool handled = false;
#ifdef MODULE_COUPLER
            handled = handled || coupler::handle(c);
#endif
#ifdef MODULE_RELAY
            handled = handled || relay::handle(c);
#endif
#ifdef MODULE_SCREWDRIVE
            handled = handled || screwdrive::handle(c);
#endif
#ifdef MODULE_T74
            handled = handled || t74::handle(c);
#endif
            (void)handled;
        }
    }

#ifdef MODULE_COUPLER
    coupler::poll(); //  checks the toolstatus periodically and sends emergency halt if needed
#endif
#ifdef MODULE_SCREWDRIVE
    screwdrive::poll(); //  ends timed runs and ramps and reports them
#endif
#ifdef MODULE_T74
    t74::poll(); //  reports move results, faults and identification samples
#endif

    //  Short: the loop reads one byte per pass, and the t74 streams identification samples
    //  from poll(). (Timed work does not depend on it -- the screwdrive is serviced from
    //  yield() and the t74 runs from its own timer interrupt.)
    delay(10);
}
