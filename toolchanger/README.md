# Toolchanger

Arduino-driven toolchanger for the UR cell: a servo that clamps the ball bearings onto a tool,
a proximity sensor that checks a tool is really there, and a relay that powers the motor.

```
toolchanger/
  toolchanger.py      pyserial driver + CLI (import ToolChanger from your UR code)
  fake_board.py       simulated board on a pty -- tests the driver with no hardware
  firmware/
    main.cpp          the Arduino Uno sketch
    build_flash.sh    compile (and optionally flash) using the ~/.arduino15 toolchain
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
| `hold` | `1` | `changeStatus(1)` -> servo to **50 deg** (locked), then checks the sensor | `1 confirmed!` or `emergency stop` |
| `release` | `0` | `changeStatus(0)` -> servo to **15 deg** (unlocked), then checks the sensor | `0 confirmed!` or `emergency stop` |
| `status` | `s` | reads the sensor **once**, changes nothing, no retry | `<n> confirmed!` or `emergency stop` |
| `probe` | `r` | reads the sensor and prints the raw number. **Moves nothing** | `raw 812 thresh 100 tool yes status 1` |
| `calibrate` | `r`, twice | prompts you to mount and remove a tool, then recommends `thresh`. **Moves nothing** | the two readings and the values to paste into `main.cpp` |
| `motor` | `m` | toggles relay K1 (`analogWrite` 201 ~ 4 V) | `Motor On` / `Motor Off` |
| `monitor` | -- | sends nothing, just prints what the board says | whatever arrives |
| *(no argument)* | -- | interactive prompt | `toolchanger> ` |

`hold`, `release` and `status` return **False** (CLI exit code 2) on `emergency stop`.

A round trip takes most of a second: `loop()` delays 200 ms, `changeServo()` blocks 500 ms, and
every sensor read averages 10 analog samples 10 ms apart.

From Python:

```python
from toolchanger import ToolChanger

with ToolChanger('/dev/ttyACM0') as tc:
    tc.hold()     # True = confirmed, False = emergency stop
```

## "the sensor disagrees with the commanded state"

That message means the board did what you asked and then the proximity sensor contradicted it:
told to hold but it reads nothing gripped, or told to release but something is still detected.
The servo moves either way -- the emergency stop is about the *sensor*, not the servo. Two
different causes, and the raw value tells them apart:

```
emergency stop raw=112 thresh=100     <- raw sits ON the threshold: MISCALIBRATED
emergency stop raw=20  thresh=100     <- raw is far from it: the tool really is not there
```

**If raw is near thresh, the threshold is wrong for this hardware.** The Uno's ADC is 10-bit
(0..1023), but this sketch was ported from a 12-bit board -- the original carried a comment
saying the no-tool reading is `4095`, which is impossible here, and it contradicted the code
about which side means "tool present". So `thresh = 100` was never a measured number. Fix it:

```bash
./toolchanger.py calibrate       # read-only, the servo does not move
```

It reads the sensor with a tool mounted and again with it removed, then prints the `thresh` and
`toolReadsHigh` to put in `firmware/main.cpp`. Reflash, and the disagreement goes away.

If `calibrate` reports the two readings are less than 50 counts apart, no threshold can work --
the sensor itself is the problem (wiring, supply, or distance to the tool).

**If it was only intermittent**, that part is already fixed. The old sketch took a single sensor
sample right after the servo moved and reported an emergency stop if it disagreed, so a tool
still seating read as a failure. `waitForTool()` now retries for up to `settleMax` (1500 ms) and
only fails if the sensor never comes around.

## Protocol

One ASCII byte per command; anything else (line endings, noise) is ignored by the board.

| byte     | effect                                                          |
|----------|-----------------------------------------------------------------|
| `0`..`9` | `changeStatus(n)` -- 0 unlocks (servo 15 deg), >0 locks (50 deg) |
| `s`      | report status without changing it, no retry                     |
| `r`      | print the raw averaged sensor reading                           |
| `m`      | toggle the motor relay K1                                       |

Replies: `<n> confirmed!`, `emergency stop raw=N thresh=M`, `raw N thresh M tool yes status n`,
`Motor On` / `Motor Off`, the informational `changed status from X to Y`, and `toolchanger ready`
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
build: 5116 bytes flash (15.6%), 399 bytes RAM.

**Check which build is on the board**: run `./toolchanger.py monitor` and press the reset button.
The current firmware prints `toolchanger ready`. If you do not see it, the board is still running
an older sketch -- and the old one never replied to `release` at all, which looks exactly like
the command "not working".

IntelliSense for the sketch comes from `.vscode/c_cpp_properties.json` at the repo root; the
absolute paths in it point at `~/.arduino15` and need editing on another machine.

## Testing without the board

```bash
./fake_board.py
```

Runs the driver against a simulated board on a pty and asserts the confirmation paths, both
emergency-stop directions, the motor toggle and the raw probe. Use it after changing the
protocol on either side.
