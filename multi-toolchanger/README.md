# Multi-toolchanger

The toolchanger Arduino from [`../toolchanger`](../toolchanger), refactored so one Uno can
drive several devices. Only the modules you choose are flashed to the board. The board announces
which ones it has, and the host script loads exactly those.
The modules are:

* **coupler**: a servo that locks the ball bearings onto a tool, plus a proximity sensor that
  checks the tool is there. It works the same as in `toolchanger/firmware/main.cpp`.
* **relay**: relay K1 (`motor` toggles it).
* **screwdrive**: the 12 V DC motor that turns the coupling lead screw, on a Cytron MD10C R3.
  It runs at variable speed in either direction, and can do timed runs and ramps.
* **t74**: the T74 tile motor on an IBT-2 H-bridge, with PID position control on an AMT10E2-V
  encoder. See [T74 tile motor](#t74-tile-motor).

```
multi-toolchanger/
  multitoolchanger.py      the CLI and interactive prompt
  fake_board.py            simulated board on a pty: tests everything with no hardware
  config/
    multitoolchanger.yaml  which modules to load; sequences that span modules
    coupler.yaml           coupler pins and coupler_* sequences
    relay.yaml             relay pin
    screwdrive.yaml        screwdrive pins, settings (max_rpm) and screwdrive_* sequences
    t74.yaml               t74 pins, settings (bandwidth, speeds, limits) and t74_* sequences
    t74_calibration.json   t74 measured state: counts per tile turn, identified motor model
  mtc/                     the framework
    board.py               serial connection, finding which board is which
    registry.py            how a module declares commands, argument kinds, settings
    config.py              loads the config and the modules it names
    pins.py                the Uno's pins, and the build-time check that modules' pins fit
    session.py             runs commands: q to stop, sequences, help, Tab completion
    modules/               one file per module: its device protocol and its commands
      general.py           help, sequence, wait, list, monitor, quit -- always loaded
      coupler.py  relay.py  screwdrive.py  t74.py
  firmware/                the Arduino Uno sketch, one .cpp per device
    main.cpp               boots the modules and passes each serial byte to the one that claims it
    coupler.cpp/.h  relay.cpp/.h  screwdrive.cpp/.h  t74.cpp/.h
    t74_control.h          the T74's control law, shared with its simulation
    test/t74_control_test.cpp   simulates that law against a motor + inertia (fake_board runs it)
    build_flash.sh         checks the selected modules' pins, compiles them, and flashes if you pass `upload`
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

* **`help`** lists the general commands and the modules loaded, with how many commands and
  sequences each has.
  * `help MODULE` lists that module's commands.
  * `help COMMAND` shows one command in full.
  * `help sequence` lists every sequence, grouped by the file it comes from.
  * `help MODULE sequence` lists just that module's sequences.
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

## Which modules are loaded

1. **Flashing picks the modules.** `firmware/build_flash.sh` compiles only the modules in the
   config's `modules:` list (or `--modules a,b`) into the firmware.
2. **The board announces them.** Every boot banner says what the firmware was built with:
   `toolchanger ready modules=coupler,screwdrive proto=5`.
3. **The script loads what the board reports (autodetect, the default).** It prints
   `detected modules: ...`. Settings and sequences still come from `config/<module>.yaml`.
   - A command for a module the board lacks is refused before anything is sent:
     `hold is a coupler command, and this board's firmware has no coupler`.
   - A cross-module sequence that needs an absent module shows as `CANNOT RUN`.

**`--no-detect`** uses the config's `modules:` list instead. If the board disagrees, it warns
with the exact difference and carries on; commands for a module the firmware lacks will time out.

**Edge cases:**
* **Old firmware:** firmware from before protocol 2 can't say what it has. The script warns and
  uses the config list.
* **Unknown module:** if the board has a module this script has no file for, it is skipped with
  a warning.
* **Protocol mismatch:** if the board reports a different `proto` than the script's `PROTOCOL`,
  the script refuses to start. That catches a board still running firmware whose commands have
  since changed shape. Bump `PROTOCOL` in both `firmware/main.cpp` and `mtc/config.py` whenever
  a command or reply changes.
* **Commands that don't connect:** `help` and `list` run without connecting, so they show the
  config's list.

## Pins

Every module's pins are set in the `pins:` block of its `config/<module>.yaml`, for example:

```yaml
# config/screwdrive.yaml
pins:
  pwm: 6        # must be a PWM pin
  dir: 7
```

**`build_flash.sh` checks the pins before it compiles anything.** It takes the selected modules
(the config's list, or `--modules`) and refuses any combination that can't work on an Uno:

* two modules (or roles) on one pin
* a pin that can't do what the role needs:
  * PWM: D3 D5 D6 D9 D10 D11
  * hardware interrupt: D2 D3
  * analog input: A0..A5
* D0 or D1, the serial link to the PC
* two modules taking over the same hardware. The coupler's Servo library and the t74's control
  loop both take Timer1, and PWM on D9/D10 is refused while either is present.
* two pin-change interrupts on one port, which would need one interrupt vector twice

It says what clashes and where, e.g. `D3 is wanted by t74.enc_b and relay.k1`, or
`screwdrive.pwm needs PWM on D9, which runs on timer1 -- and t74 has taken timer1 over`. Then
it compiles the pins in as `-DPIN_<MODULE>_<ROLE>`. Pins are a build-time matter only: the
script never needs them at run time.

For example, the t74 and the screwdrive share one board if the screwdrive moves to the free
Timer2 PWM pin:

```yaml
# config/screwdrive.yaml, in that board's config directory
pins: {pwm: 11, dir: 12}
```

Different boards wired differently get different config directories (`--config`), each with its
own module yamls.

## Configuration

[`config/multitoolchanger.yaml`](config/multitoolchanger.yaml) lists the modules this
toolchanger has. That list is what gets flashed, and what loads with `--no-detect`:

```yaml
modules:
  - coupler
  - relay
  - screwdrive
sequences: {}        # sequences that use more than one module
```

* Each module loads `mtc/modules/<name>.py`.
* Each also reads `config/<name>.yaml` from the same directory, if that file exists.
* A module that isn't loaded does not exist for the script. Its commands are not in `help`,
  Tab completion or the CLI.
* `general` is always loaded and is not listed.

To deploy to a different toolchanger, copy `config/` and edit it. Then pass
`--config path/to/multitoolchanger.yaml` (to the script and to `build_flash.sh`) or set
`$MULTITOOLCHANGER_CONFIG`.

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
* **Pins** are declared with `MODULE.pin(role, default, needs=...)`, where `needs` is
  `'digital'`, `'pwm'`, `'analog'`, `'interrupt'` or `'pcint'`. Hardware it takes over whole is
  declared with `MODULE.claim('timer1')`. The yaml `pins:` block moves the pins, and the build
  checks them.
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

Then add the module's name to `config/multitoolchanger.yaml`, and reflash. `help`, the CLI,
completion and sequences pick it up automatically. The module's Python file name must match its
firmware file name (`relay.py` and `relay.cpp`); that shared name is how the board's
announcement maps to a module.

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

## T74 tile motor

PID position control of the T74, which turns the tile, on an AMT10E2-V encoder mounted on the
motor shaft. The encoder turns about 20.3 times per tile turn, giving 40,000 counts per tile
turn at 500 PPR. This replaces `t74-motor/` and its stop-early-and-correct moves.

### Wiring

The default pins in `config/t74.yaml` are `t74-motor/`'s wiring:

| T74 board | Uno |
|---|---|
| IBT-2 RPWM / LPWM | D5 (forward) / D6 (reverse) |
| IBT-2 R_EN / L_EN | D7 / D8 (fit 10 kΩ pull-downs: the pins float through a reset) |
| IBT-2 VCC / GND | 5V / GND |
| encoder A / B / X (index) | D2 / D3 / D4 |
| encoder 5V / G | 5V / GND |

**Pin constraints.** The t74 counts every encoder edge, so A and B need the Uno's only two
hardware interrupts (D2 and D3). Its 500 Hz control loop takes Timer1 over. So it can never
share a board with the coupler, whose Servo library also needs Timer1. With moved pins, the
screwdrive *can* share its board (see [Pins](#pins)). With the defaults, flash it on its own:

```bash
cd firmware && ./build_flash.sh upload --modules t74 --port /dev/ttyACMx
```

Set the encoder's DIP switches to 500 PPR (`0 1 0 1`); 5120 PPR is too fast for the Uno.
`t74_watch`, with the motor supply off and the shaft turned by hand, checks the wiring: the count
should change smoothly, missed edges should stay 0, and the index should pulse once per encoder
turn.

### Setting it up

1. **`t74_calibrate [TURNS]`**: counts the encoder counts per tile turn while you watch a mark
   on the tile. This is needed once per gearbox. It carries over the -40000 counts from
   `t74-motor/`, so you can skip it unless you have changed the gearbox. Or set the count
   directly with `t74_counts N`.
2. **`t74_identify` with the HEAVIEST load mounted.**
   - What it does: steps the PWM to `identify_pwm1`, then `identify_pwm2` (about half a tile
     turn forward), and fits the motor model to the encoder.
   - What it measures: the speed per PWM (K), the mechanical time constant (τ, which grows
     with the load's inertia) and the gearbox friction.
   - Where it goes: the results are saved to `config/t74_calibration.json`, and the gains are
     computed and sent.
3. **`t74_move 90`** four times, or `sequence t74_quarter_turns`, should bring the mark back to
   the reference. Each move prints its final error.

After that, every connect re-sends the limits and gains. The board forgets them on reset, and
opening the port resets it.

### Moving

| command | does |
|---|---|
| `t74_move DEG` | turn DEG degrees from the current target (so a run of moves doesn't add up errors) |
| `t74_goto DEG` | turn to DEG degrees from the zero, the shortest way round (never more than half a turn, crossing the zero if that is shorter) |
| `t74_home` | run forward to the encoder index and make it zero (a repeatable *motor* position: with the encoder on the motor side, the index repeats about every 18° of tile) |
| `t74_zero` | make here 0° |
| `t74_halt` | brake to a stop and hold there |
| `t74_hold [on\|off]` | toggle (or set) holding after each move, for this session |
| `t74_stop` | release the motor |
| `t74_status`, `t74_watch` | position, target, error, PWM, encoder health |
| `t74_tune [BANDWIDTH]` | show the gains, or recompute them for another bandwidth this session |

* **Every move follows a trapezoidal profile** (`max_speed_dps`, `max_accel_dps2`), and ends
  holding the target. Holding keeps an unbalanced load from back-driving the tile.
* **Holding is configurable.** Set `hold_after_move: false` in `config/t74.yaml`, or run
  `t74_hold off` for a session. The board then releases the motor as soon as each move settles,
  which means less heat, but the load can drift. A relative move still counts from the last
  target. With holding off, a one-shot CLI move just exits rather than holding the connection
  open.
* **q halts and holds; `t74_stop` releases.** An unbalanced load is never dropped because
  someone pressed q.
* **Closing the port releases the motor**, because the reset is a dead-man switch. So a one-shot
  `./multitoolchanger.py t74_move 90` keeps the connection open, holding, until q or Ctrl-C.
  `--latch` leaves the board running.
* **The board cuts the motor and reports a FAULT** if:
  * the following error exceeds `max_error_deg` (jammed, overloaded, or a wrong-sign model)
  * it gets no encoder counts for 1 s at full PWM
  * homing finds no index within 1.25 encoder turns

### How it is tuned

Model: `τ·y'' + y' = K·(u − friction)`, with y in encoder counts and u in PWM. The controller is

    u = (v_ref + τ·a_ref)/K  +  friction·sign  +  Kp·e + Ki·∫e + Kd·ė

**Feedforward.** The first term is the model inverted along the motion profile. With a perfect
model the error stays zero, and the feedback only corrects mistakes and disturbances.

**Friction compensation.** This is essential. Without it, the integral has to wind up past the
gearbox friction before the motor moves, then jumps past the target and hunts. In simulation an
uncompensated motor sticks 11 to 12 counts off target.

**Pole placement.** The PID gains put all three closed-loop poles at `−bandwidth`:

    Kp = 3τω²/K    Ki = τω³/K    Kd = (3τω − 1)/K

That is critically damped at the identified load. The integral holds an unbalanced load at the
target. A sudden load shift while holding is pushed back without crossing the target.

**Why identify at the heaviest load.** The triple pole is the best-damped point, not a floor that
holds for every load. A heavier load than identified rings. A lighter one tends to a damping
ratio of √3/2 ≈ 0.87, a fraction of a percent of overshoot.

`firmware/test/t74_control_test.cpp` runs the firmware's exact control law (`t74_control.h`)
against a simulated motor and inertia. The simulation includes encoder quantization, PWM
saturation, friction and an unbalanced load, and checks:

* no overshoot beyond the in-position band at the tuned load
* at most about 0.2° at a third of its inertia
* no hunting with friction, even with the friction estimate a third out either way

`./fake_board.py` builds and runs it.

## Protocol

| bytes | module | effect | reply |
|---|---|---|---|
| `?` | main | repeat the identity line | `modules=<a,b> proto=<n>` |
| `0`..`9` | coupler | 0 unlocks (servo 15 deg), >0 locks (50 deg) | `<n> confirmed!` / `emergency stop raw=N thresh=M` |
| `s` | coupler | status, no retry | `tool present/absent`, then confirm or emergency |
| `r` | coupler | raw sensor reading | `raw N thresh M tool yes status n bypass off` |
| `b` | coupler | toggle sensor bypass (any reset clears it) | `sensor bypass ON ...` / `off` |
| `m` | relay | toggle K1 | `Motor On` / `Motor Off` |
| `d<pct>\n` | screwdrive | signed speed -100..100, 0 stops | `screwdrive <pct>%` |
| `t<pct>,<ms>\n` | screwdrive | the same for ms (1..3600000), then stop | `screwdrive <pct>% for <ms> ms`, later `screwdrive run done` |
| `p<duty>,<ms>\n` | screwdrive | raw duty -255..255 for ms, then stop (`pwm`, and `rpm` after conversion) | `screwdrive pwm <duty> for <ms> ms`, later `screwdrive run done` |
| `J...` `L...` `R<counts>` `A<counts>,<turn>` `H` `Z` `S` `X` `O<pwm>` `I<p1>,<p2>,<ms>` `E` | t74 | gains, limits, moves, home, zero, halt, release, open run, identify, report (see `mtc/modules/t74.py`) | `t74 ...` lines; `t74 rejected: ...` or `t74 FAULT ...` on trouble |
| `a<p1>,<p2>,<ms>\n` | screwdrive | ramp p1% to p2% over ms, then hold p2% | `screwdrive ramp <p1>% to <p2>% over <ms> ms`, later `screwdrive ramp done` |

The driver rejects out-of-range values before it sends anything, and the board also clamps
them. If the board receives a malformed line, it replies `screwdrive rejected a malformed ...,
stopping` and stops the motor. It always reads to the end of the line, so none of the bad
characters can be run as a separate command. That matters because a stray digit would be a
coupler hold or release.

The board prints `toolchanger ready modules=<a,b> proto=<n>` on every boot. The link runs at
**115200 baud** since protocol 3 (now 5). Before that it was 9600, so every board needs reflashing once:
the script warns about a board it gets no boot banner from.

## Adding a device to the firmware

Every firmware module has the same three parts: `setup()`, `handle(c)` (returns true if it
claimed the byte), and `poll()` if it needs periodic work. To add a device:

1. Write `firmware/<name>.cpp/.h`. `build_flash.sh` treats every `.cpp` other than `main` as a
   module. Use `PIN_<NAME>_<ROLE>` for every pin, with an `#ifndef ... #error` guard like the
   existing modules. Declare those pins in the module's Python file, so the build can check
   them.
2. In `main.cpp`, add `#ifdef MODULE_<NAME>` blocks for its include, `setup()`, `handle()`
   chain entry and `poll()`.
3. Add its name to `printIdentity()`.

Command bytes must not clash. The ones in use are listed at the top of `main.cpp`. Then add the
host side (see [Adding a module](#adding-a-module)).

## Build and test

```bash
cd firmware
./build_flash.sh                                    # build the config's modules; touches no hardware
./build_flash.sh upload --port /dev/ttyACM2         # build and flash them
./build_flash.sh upload --modules screwdrive --port /dev/ttyACM2   # just these, ignoring the config
./build_flash.sh --config path/to/multitoolchanger.yaml            # another deployment's list
cd .. && ./fake_board.py                            # driver vs. a simulated board
```

From /ur-assembly/:
```bash
./multi-toolchanger/firmware/build_flash.sh upload --modules <module> --port /dev/ttyACMX
```


Flash size depends on the modules. Coupler, relay and screwdrive together are about 7.8 KB
(23.7%), screwdrive alone 5.6 KB, and t74 13.6 KB (it carries floating-point control). The Servo library is compiled in only with the coupler. `build_flash.sh` defaults to
`PORT=/dev/ttyACM1`, so pass `--port` (or set `PORT`) for the board you mean. If you flash the
coupler, its servo moves when the board resets.
