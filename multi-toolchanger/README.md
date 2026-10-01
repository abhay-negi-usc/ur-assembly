# Multi-toolchanger

The toolchanger Arduino from [`../toolchanger`](../toolchanger), refactored so one Uno can
drive several devices, with a host script that loads only the devices a given toolchanger has.
The modules are:

* **coupler**: a servo that locks the ball bearings onto a tool, plus a proximity sensor that
  checks the tool is there. It works the same as in `toolchanger/firmware/main.cpp`.
* **relay**: relay K1 (`motor` toggles it).
* **screwdrive**: the 12 V DC motor that turns the coupling lead screw, on a Cytron MD10C R3.
  It runs at variable speed in either direction, and can do timed runs and ramps.

```
multi-toolchanger/
  multitoolchanger.py      the CLI and interactive prompt
  fake_board.py            simulated board on a pty: tests everything with no hardware
  config/
    multitoolchanger.yaml  which modules to load; sequences that span modules
    coupler.yaml           coupler settings and coupler_* sequences
    screwdrive.yaml        screwdrive settings (max_rpm) and screwdrive_* sequences
  mtc/                     the framework
    board.py               serial connection, finding which board is which
    registry.py            how a module declares commands, argument kinds, settings
    config.py              loads the config and the modules it names
    session.py             runs commands: q to stop, sequences, help, Tab completion
    modules/               one file per module: its device protocol and its commands
      general.py           help, sequence, wait, list, monitor, quit -- always loaded
      coupler.py  relay.py  screwdrive.py
  firmware/                the Arduino Uno sketch, one .cpp per device
    main.cpp               boots the modules and passes each serial byte to the one that claims it
    coupler.cpp/.h  relay.cpp/.h  screwdrive.cpp/.h
    build_flash.sh         compiles every *.cpp, and flashes if you pass `upload`
```

## Screwdrive wiring

The 12 V supply goes straight to the MD10C's power terminals and the motor goes on its motor
terminals. The MD10C makes its own logic supply from the 12 V, so the Arduino only needs three
wires to it:

| MD10C | Arduino | |
|---|---|---|
| PWM | D6 | speed. Timer0, about 980 Hz (the MD10C accepts up to 20 kHz) |
| DIR | D7 | LOW = forward (positive %), HIGH = reverse |
| GND | GND | common reference. **Required.** |

* **Fit a 10 kΩ pull-down from PWM to GND.** Opening the serial port resets the Uno, and the
  pins float through the reset and the bootloader until `setup()` drives them low.
* Why these pins: the Servo library takes Timer1, which disables PWM on 9 and 10. Pin 3 is the
  relay and pin 5 is the servo. The pins are constants at the top of `screwdrive.cpp`.
* If "forward" turns the wrong way, swap the motor leads.

## Usage

```bash
./multitoolchanger.py --port dc_motor                  # interactive prompt
./multitoolchanger.py help                             # every loaded command, by module
./multitoolchanger.py help ramp                        # one command in full
./multitoolchanger.py ramp 0 60 3 --port dc_motor      # 0% -> 60% over 3 s, then hold 60%
./multitoolchanger.py sequence screwdrive_attach --port dc_motor
```

At the prompt:

* **`help`** lists every command under the module it belongs to, then the sequences, grouped
  by the file each came from. `help COMMAND` shows one command in full, and `help MODULE` lists
  one module with its sequences.
* **Tab** completes command names, sequence names after `sequence`, and topics after `help`.
* **Up/down arrows** go through previous commands. History is kept in
  `~/.multitoolchanger_history`.
* **q** stops a timed command (`run`, `pwm`, `rpm`, `ramp`, `wait`) or a sequence at any point.
  It stops anything moving (each module decides what that means; for the screwdrive it is the
  motor) and abandons the command. The prompt then carries on as normal; it is not an emergency
  state. Ctrl-C does the same.
* Commands must be typed in full. There are no shorthands; use Tab instead.

`--port` takes the names in `configs/couplers.yaml`, a USB serial, or a device path.

From Python:

```python
import sys; sys.path.insert(0, 'multi-toolchanger')
from mtc import ToolChanger

with ToolChanger('dc_motor') as tc:   # loads config/multitoolchanger.yaml
    tc.screwdrive.run(40, 2.5)        # blocks until the board says the run is over
    tc.screwdrive.ramp(0, 60, 3)      # then holds 60%
    tc.screwdrive.stop()
    tc.coupler.hold()
```

## Configuration

**The config decides which modules exist.** [`config/multitoolchanger.yaml`](config/multitoolchanger.yaml)
lists the modules this toolchanger has:

```yaml
modules:
  - coupler
  - relay
  - screwdrive
sequences: {}        # sequences that use more than one module
```

* Each listed module loads `mtc/modules/<name>.py`.
* Each also reads `config/<name>.yaml` from the same directory, if that file exists.
* A module that isn't listed does not exist for the script. Its commands are not in `help`,
  Tab completion or the CLI.
* `general` is always loaded and is not listed.

To deploy to a different toolchanger, copy `config/` and edit it. Then pass
`--config path/to/multitoolchanger.yaml` or set `$MULTITOOLCHANGER_CONFIG`.

Each module's file holds its settings and its own sequences:

```yaml
# config/screwdrive.yaml
settings:
  max_rpm: 500       # no-load rpm at full duty; `rpm` scales from this
sequences:
  screwdrive_attach:
    description: Drive the coupling screw in until bottomed out
    steps:
      - run 100 18.5
      - wait 1
      - run 15 20
```

**What stops the script from starting:**
* a missing or unreadable main file
* an unknown module
* two modules defining the same command
* an unknown or bad setting

Falling back to defaults could quietly run with the wrong modules or numbers.

### Sequences

* **Naming:** sequences in `config/<module>.yaml` must be named `<module>_<name>`, for example
  `screwdrive_attach` or `coupler_tool_cycle`. That way you can always tell where one came from.
* **Which commands a step can use:** a module's sequences may use only that module's commands,
  plus `wait`. Sequences that span modules, such as `hold` then `run`, go in
  `multitoolchanger.yaml`, where any loaded command is allowed.
* **Writing steps:** each step is a command written exactly as you'd type it at the prompt. A
  sequence can be a bare list of steps, or a mapping with `description:` and `steps:`.
* **Checking:** every step is checked when the config loads. A sequence with a problem is
  listed by `help` as `CANNOT RUN`, and `sequence NAME` says what's wrong without running any of
  it. One bad sequence never stops the others loading.
* **Failures:** if a step fails, the rest are skipped and anything moving is stopped. For
  example, a `hold` that finds no tool, or an emergency stop from the coupler watchdog during a
  `wait`.
* **Motor left running:** `drive` and `ramp` leave the motor running into the next step. Finish
  with `stop`, or a ramp down to 0.
* **Not allowed in a sequence:** another sequence, `calibrate`, or other interactive commands.

## Adding a module

A module is one file in `mtc/modules/` holding its device protocol and its commands.
`relay.py` is the shortest example:

```python
from ..registry import Module

class Relay:                              # built once per connection
    def __init__(self, board, settings):
        self.board = board
    def toggle(self):
        final, _ = self.board.exchange('m', lambda ln: ln.startswith('Motor'))
        return final == 'Motor On'

MODULE = Module('relay', 'relay K1 (firmware/relay.cpp)', device=Relay)

@MODULE.command()
def cmd_motor(s):
    """Toggle relay K1 on or off.

    Longer explanation, shown by `help motor`."""
    s.dev('relay').toggle()
```

* **The docstring is the help text.** Its first line is the summary in `help`, and the rest is
  shown by `help <command>`. A command without a docstring won't load.
* **Arguments** are `Param(NAME, kind)`.
  * Built-in kinds: `seconds`, `sequence`, `topic`.
  * A module can add its own with `MODULE.kind(...)`, as the screwdrive does for `percent`,
    `duty` and `rpm`.
  * A kind validates the value, describes it in help, and can supply Tab completion.
* **Settings** are declared with `MODULE.setting(name, default, check)` and read from the
  module's yaml `settings:` block.
* **Command flags:**
  * `timed=True`: q stops it.
  * `in_sequence=False`: not allowed in sequences.
  * `board=False`: runs without a connection.
  * `runs_on=f`: it leaves something running, so the one-shot CLI holds the port open.
* **A handler returns False to report failure.** That aborts a sequence.
* **Optional device hooks:**
  * `safe_stop()`: called on q or a failed sequence.
  * `check_line(line)`: raises on a message the board sends unprompted that should fail a
    wait.

Then add the module's name to `config/multitoolchanger.yaml`. `help`, the CLI, completion and
sequences pick it up automatically.

The firmware side is a matching `.cpp` module; see [Adding a device to the firmware](#adding-a-device-to-the-firmware).

## Screwdrive: timed runs, ramps and rpm

The **board** times `run`, `pwm`, `rpm` and `ramp`, not the host. It stops the motor when the
time is up (or, for a ramp, settles at the end speed) and reports that it's finished. A host
that hangs mid-run can't leave the motor on a timer that never ends. Any new screwdrive command
replaces a run or ramp in progress.

**`ramp START END SECONDS`** starts at START% at once and changes speed linearly to END% over
SECONDS. Then it **keeps running at END**, like `drive`. Ramp to 0 to finish stopped. Crossing
zero, for example from 50 to -50, passes through a stop rather than braking.

The timing stays accurate while the coupler is busy. The coupler's servo moves and sensor reads
block for up to about 2 s, but `delay()` calls `yield()` throughout, and `main.cpp` uses
`yield()` to step ramps and end runs on time.

**`rpm` is open loop.** The host converts it to a duty, rpm / max_rpm × 255, using `max_rpm`
from `config/screwdrive.yaml`, and sends a timed `pwm` run. The firmware has no rpm figure of
its own. Nothing measures the real speed, so:

* under load the motor turns slower than asked
* at low rpm (low duty) it may not start at all, because of static friction
* if the supply is not 12 V, the speeds scale with it

**By default, closing the port stops the motor.** Closing it resets the board, and `setup()`
drives PWM low. That makes it a dead-man switch: a crashed script cannot leave the motor
running. So when a one-shot `drive` or `ramp` leaves the motor running, the CLI holds the
connection open until you press q or Ctrl-C. Pass `--latch` to give that up.

If the motor is running and you reverse it with `drive`, the board cuts the drive for 200 ms
(`reverseDwellMs`) before it flips DIR. On the MD10C, PWM LOW shorts both motor terminals to
ground, so the motor is braked during that time rather than left to coast.

## Protocol

| bytes | module | effect | reply |
|---|---|---|---|
| `0`..`9` | coupler | 0 unlocks (servo 15 deg), >0 locks (50 deg) | `<n> confirmed!` / `emergency stop raw=N thresh=M` |
| `s` | coupler | status, no retry | `tool present/absent`, then confirm or emergency |
| `r` | coupler | raw sensor reading | `raw N thresh M tool yes status n bypass off` |
| `b` | coupler | toggle sensor bypass (any reset clears it) | `sensor bypass ON ...` / `off` |
| `m` | relay | toggle K1 | `Motor On` / `Motor Off` |
| `d<pct>\n` | screwdrive | signed speed -100..100, 0 stops | `screwdrive <pct>%` |
| `t<pct>,<ms>\n` | screwdrive | the same for ms (1..3600000), then stop | `screwdrive <pct>% for <ms> ms`, later `screwdrive run done` |
| `p<duty>,<ms>\n` | screwdrive | raw duty -255..255 for ms, then stop (`pwm`, and `rpm` after conversion) | `screwdrive pwm <duty> for <ms> ms`, later `screwdrive run done` |
| `a<p1>,<p2>,<ms>\n` | screwdrive | ramp p1% to p2% over ms, then hold p2% | `screwdrive ramp <p1>% to <p2>% over <ms> ms`, later `screwdrive ramp done` |

The driver rejects out-of-range values before it sends anything, and the board also clamps
them. If the board receives a malformed line, it replies `screwdrive rejected a malformed ...,
stopping` and stops the motor. It always reads to the end of the line, so none of the bad
characters can be run as a separate command. That matters because a stray digit would be a
coupler hold or release.

The board prints `toolchanger ready` on every boot.

## Adding a device to the firmware

Every firmware module has the same three parts: `setup()`, `handle(c)` (returns true if it
claimed the byte), and `poll()` if it needs periodic work. To add a device:

1. Write the new module.
2. Call its `setup()` in `main.cpp`.
3. Add it to the `||` chain in `loop()`.

Command bytes must not clash. The ones in use are listed at the top of `main.cpp`. Then add the
host side (see [Adding a module](#adding-a-module)).

## Build and test

```bash
cd firmware
./build_flash.sh                            # build only, touches no hardware
PORT=/dev/ttyACM2 ./build_flash.sh upload   # flash it (the servo WILL move on reset)
cd .. && ./fake_board.py                    # driver vs. a simulated board
```

The current build is 7626 bytes of flash (23.3%) and 736 bytes of RAM. `build_flash.sh`
defaults to `PORT=/dev/ttyACM1`, so set `PORT` to the board you mean.
