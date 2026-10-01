# Toolchanger

Arduino-driven toolchanger for the UR cell: a servo that clamps the ball bearings onto a tool,
a proximity sensor that checks a tool is really there, a relay that powers the motor, and a
12 V DC motor on a Cytron MD10C R3 driver (variable speed, both directions).

```
toolchanger/
  toolchanger.py      pyserial driver + CLI (import ToolChanger from your UR code)
  fake_board.py       simulated board on a pty -- tests the driver with no hardware
  firmware/           the Arduino Uno sketch, one module per device
    main.cpp          boots the modules and routes each serial byte to the one that claims it
    coupler.cpp/.h    servo + proximity sensor: hold/release/status/probe/bypass, watchdog
    relay.cpp/.h      motor relay K1 ('m')
    dc_motor.cpp/.h   MD10C DC motor driver ('d<pct>')
    build_flash.sh    compile every *.cpp (and optionally flash) using the ~/.arduino15 toolchain
```

Run everything with a leading `./` -- the shell does not search the current directory:

```bash
cd toolchanger
./toolchanger.py --help          # NOT `toolchanger.py`, that is "command not found"
```

## What each command actually does

Every one of these **opens the serial port, which pulls DTR and resets the Arduino**, so the
board reboots and the servo swings to its boot position before your command is even sent. The
driver waits `--settle` seconds (default 2) for the bootloader, then sends a single ASCII byte.

| command | byte | what the board does | what you get back |
|---|---|---|---|
| `hold` | `1` | `changeStatus(1)` -> servo to **50 deg** (locked). Moves even with nothing to grip, and says so | `1 confirmed!` or `locked, no tool detected` |
| `release` | `0` | `changeStatus(0)` -> servo to **15 deg**, bearings retract. Always confirms | `0 confirmed!`, after `released, tool ...` |
| `status` | `s` | reads the sensor **once**, changes nothing, no retry | `tool present`/`absent`, then a confirm or an alarm |
| `probe` | `r` | reads the sensor and prints the raw number. **Moves nothing** | `raw 812 thresh 100 tool yes status 1` |
| `calibrate` | `r`, twice | prompts you to mount and remove a tool, then recommends `thresh`. **Moves nothing** | the two readings and the values to paste into `coupler.cpp` |
| `bypass` | `b` | stops the board consulting the sensor at all. **Moves nothing** | `sensor bypass ON` / `off` |
| `motor` | `m` | toggles relay K1 (`analogWrite` 201 ~ 4 V) | `Motor On` / `Motor Off` |
| `drive N` | `dN\n` | DC motor at N% (-100..100, negative reverses, 0 stops) | `dc motor N%` |
| `stop` | `d0\n` | DC motor off | `dc motor 0%` |
| `monitor` | -- | sends nothing, just prints what the board says | whatever arrives |
| *(no argument)* | -- | interactive prompt | `toolchanger> ` |

`hold` returns **False** (CLI exit code 2) when it locked onto nothing, and `status` on
`emergency stop` -- a gripped tool that has gone. `release` confirms
whenever the servo was commanded -- see below for why it is not symmetric with `hold`.

A round trip takes most of a second: `loop()` delays 200 ms, `changeServo()` blocks 500 ms, and
every sensor read averages 10 analog samples 10 ms apart.

From Python:

```python
from toolchanger import ToolChanger

with ToolChanger('/dev/ttyACM0') as tc:
    tc.hold()     # True = tool gripped, False = locked but nothing there
```

## Working the servo with a broken sensor

A probe that is unplugged, unpowered or shorted does not read as "no tool" -- it reads a hard
rail, 0 or 1023, and then *agrees with everything*. That is worse than no sensor at all,
because the board goes on confirming grips that are not happening. When the readings look like
that (`probe` returning a dead-stable 0 or 1023 no matter what is in the changer), bypass it:

```bash
./toolchanger.py            # the bypass only lasts as long as the connection
toolchanger> bypass         # "sensor bypass ON -- grip is NOT verified"
toolchanger> hold           # servo moves, confirms, says "(sensor bypassed)"
toolchanger> release
```

Bypassed, `hold` and `release` move the servo and confirm without asking the sensor, and the
10 s watchdog stops alarming. **Grip is not verified** -- every confirmation means only "the
servo was commanded".

It lives in RAM and any reset clears it, and opening the serial port resets the board, so a
bypass can never outlive the connection that set it. You cannot leave the machine bypassed by
accident.

## Hold and release are not mirror images

The proximity sensor answers one question: **is a tool present?** That is not the same question
as "are the bearings clamped", and treating it as if it were is what made `release` fail.

* **Holding** onto nothing is a **notice**, not a fault. The servo moves either way; if the
  sensor then sees no tool the board says `locked, no tool detected raw=N thresh=M` instead of
  `confirmed!`. `hold()` returns False for it -- a pick that missed must not lift -- but nothing
  alarms, and the watchdog stays quiet, so a coupler can be locked empty on purpose.
* **Releasing** retracts the bearings, but the tool goes on sitting in the changer until
  something physically pulls it away. The sensor still seeing it is the **normal** outcome, so
  a release confirms and reports `released, tool still in the changer`. An earlier version
  demanded an empty reading here and turned every good release into an emergency stop.
* The genuinely dangerous case -- a tool that WAS gripped has since fallen out -- is still an
  `emergency stop`, from the `checkTime()` watchdog every 10 s and from `status`. The board
  tracks this as `toolHeld`: set only when a lock saw a tool, cleared on release.

## "the sensor disagrees with the commanded state"

An `emergency stop` now means a gripped tool has gone; `locked, no tool detected` means a
hold found nothing. Both carry the reading, and for both there are two different causes that
the raw value tells apart:

```
locked, no tool detected raw=112 thresh=100   <- raw sits ON the threshold: MISCALIBRATED
locked, no tool detected raw=20  thresh=100   <- raw is far from it: the tool really is not there
```

**If raw is near thresh, the threshold is wrong for this hardware.** The Uno's ADC is 10-bit
(0..1023), but this sketch was ported from a 12-bit board -- the original carried a comment
saying the no-tool reading is `4095`, which is impossible here, and it contradicted the code
about which side means "tool present". So `thresh = 100` was never a measured number. Fix it:

```bash
./toolchanger.py calibrate       # read-only, the servo does not move
```

It reads the sensor with a tool mounted and again with it removed, then prints the `thresh` and
`toolReadsHigh` to put in `firmware/coupler.cpp`. Reflash, and the disagreement goes away.

If `calibrate` reports the two readings are less than 50 counts apart, no threshold can work --
the sensor itself is the problem (wiring, supply, or distance to the tool).

**If it was only intermittent**, that part is already fixed. The old sketch took a single sensor
sample right after the servo moved and reported an emergency stop if it disagreed, so a tool
still seating read as a failure. `waitForTool()` now retries for up to `settleMax` (1500 ms) and
only fails if the sensor never comes around.

## Protocol

One ASCII byte per command -- except `d`, which is followed by a number -- and anything else
(line endings, noise) is ignored by the board.

| byte     | effect                                                          |
|----------|-----------------------------------------------------------------|
| `0`..`9` | `changeStatus(n)` -- 0 unlocks (servo 15 deg), >0 locks (50 deg) |
| `s`      | report status without changing it, no retry                     |
| `r`      | print the raw averaged sensor reading                           |
| `b`      | toggle the sensor bypass (cleared by any reset)                 |
| `m`      | toggle the motor relay K1                                       |
| `d<pct>\n` | DC motor speed, signed percent -100..100; 0 or a garbled number stops |

Replies: `<n> confirmed!`, `locked, no tool detected raw=N thresh=M`, `emergency stop raw=N thresh=M`, `raw N thresh M tool yes status n`,
`Motor On` / `Motor Off`, `dc motor <pct>%`, the informational `changed status from X to Y`, and `toolchanger ready`
on boot.

## Firmware

```bash
cd firmware
./build_flash.sh            # build only -- touches no hardware
./build_flash.sh upload     # build then flash (the servo WILL move on reset)
PORT=/dev/ttyACM1 ./build_flash.sh upload
```

Builds against the toolchain the Arduino IDE installs under `~/.arduino15` (avr-gcc 7.3, Servo,
avrdude) -- no PlatformIO, no IDE. Target is an Uno / ATmega328P at 16 MHz, 9600 baud. Current
build: 6030 bytes flash (18.4%), 561 bytes RAM.

Each device is a module with the same shape -- `setup()`, `handle(byte)` returning whether it
claimed the byte, and `poll()` where it needs one -- and `main.cpp` only boots them and routes
bytes. Adding a device is a new module, a `setup()` call and one more `||` in `loop()`.

**Check which build is on the board**: run `./toolchanger.py monitor` and press the reset button.
The current firmware prints `toolchanger ready`. Two other quick tells that you are on an older
sketch: `probe` times out (nothing implements `'r'`), and an emergency stop comes back as a bare
`emergency stop` with no `raw=N thresh=M` after it.

IntelliSense for the sketch comes from `.vscode/c_cpp_properties.json` at the repo root; the
absolute paths in it point at `~/.arduino15` and need editing on another machine.

## DC motor (Cytron MD10C R3)

The motor and the MD10C run off an external 12 V supply (the MD10C regulates its own logic
from it). The Arduino needs three wires:

| MD10C | Arduino | |
|---|---|---|
| PWM | D6 | speed; Timer0, ~980 Hz (the MD10C takes up to 20 kHz) |
| DIR | D8 | LOW = forward (positive %), HIGH = reverse |
| GND | GND | common reference -- required |

```bash
./toolchanger.py drive 40 --port cleat     # forward at 40%
./toolchanger.py drive -40 --port cleat    # reverse
./toolchanger.py stop --port cleat
```

* **Fit a 10 kΩ pull-down from PWM to GND.** Opening the serial port resets the board, and
  through reset and the bootloader the Uno's pins float until `setup()` drives them low.
* **Closing the port stops the motor**, because the reset runs `setup()` again -- a dead-man
  switch. From the CLI a `drive` therefore lasts only as long as the command, so use the
  interactive prompt, or `--latch` to leave it running (then `stop` it explicitly).
* Reversing while running cuts the drive and coasts 200 ms (`reverseDwellMs`) before flipping
  DIR, so the motor is never plugged straight into full reverse.
* Swap the motor leads if "forward" turns the wrong way. Pins are constants in `dc_motor.cpp`;
  avoid 9/10 (the Servo library takes Timer1) and 3/5 (relay, servo).
* Out-of-range values (`drive 600`) are refused by the driver before the port is opened; the
  board clamps anyway, and a garbled number parses as 0 -- it fails stopped.

## Testing without the board

```bash
./fake_board.py
```

Runs the driver against a simulated board on a pty and asserts the confirmation paths, both
empty-lock notice, the lost-tool emergency stop, the motor toggle, the DC motor speeds and the raw probe. Use it after changing the
protocol on either side.
