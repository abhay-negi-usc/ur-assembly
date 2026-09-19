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

## Driving it from the host

```bash
./toolchanger.py hold       # clamp onto the tool      (sends '1')
./toolchanger.py release    # let the tool go          (sends '0')
./toolchanger.py status     # is a tool really there?  (sends 's')
./toolchanger.py motor      # toggle the motor relay   (sends 'm')
./toolchanger.py monitor    # watch what the board prints
./toolchanger.py            # interactive prompt
```

The port is autodetected; override with `--port /dev/ttyACM0`. From Python:

```python
from toolchanger import ToolChanger

with ToolChanger('/dev/ttyACM0') as tc:
    tc.hold()     # True = confirmed, False = emergency stop
```

`hold()` / `release()` / `status()` return **False** when the board answers `emergency stop`,
which means the proximity sensor disagrees with the commanded state -- asked to hold but nothing
is gripped, or asked to release but something is still detected. The CLI exits 2 in that case.

**Opening the port resets the Arduino** (DTR), and the servo moves on boot. The driver waits
`--settle` seconds (default 2) for the bootloader before sending. A round trip takes most of a
second: `loop()` delays 200 ms, `changeServo()` blocks 500 ms, and every sensor read averages
10 analog samples 10 ms apart.

## Protocol

One ASCII byte per command; anything else (line endings, noise) is ignored by the board.

| byte     | effect                                                      |
|----------|-------------------------------------------------------------|
| `0`..`9` | `changeStatus(n)` -- 0 unlocks (servo 15 deg), >0 locks (50 deg) |
| `s`      | report status without changing it                            |
| `m`      | toggle the motor relay K1                                    |

Replies: `<n> confirmed!`, `emergency stop`, `Motor On` / `Motor Off`, and the informational
`changed status from X to Y`.

## Firmware

```bash
cd firmware
./build_flash.sh            # build only -- touches no hardware
./build_flash.sh upload     # build then flash (the servo WILL move on reset)
PORT=/dev/ttyACM1 ./build_flash.sh upload
```

Builds against the toolchain the Arduino IDE installs under `~/.arduino15` (avr-gcc 7.3,
Servo, avrdude) -- no PlatformIO, no IDE. Target is an Uno / ATmega328P at 16 MHz, 9600 baud.
Current build: 4818 bytes flash (14.7%), 327 bytes RAM.

IntelliSense for the sketch comes from `.vscode/c_cpp_properties.json` at the repo root; the
absolute paths in it point at `~/.arduino15` and will need editing on another machine.

## Testing without the board

```bash
./fake_board.py
```

Runs the driver against a simulated board on a pty and asserts both confirmation paths and both
emergency-stop paths. Use it after changing the protocol on either side.
