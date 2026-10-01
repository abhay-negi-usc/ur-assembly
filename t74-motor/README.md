# T74 motor: angle control with an AMT10E2-V encoder

An Arduino Uno drives the T74 DC motor through an IBT-2 (BTS7960) H-bridge. An AMT10E2-V
quadrature encoder on the **motor shaft, before the gearbox** (about 20.3 encoder turns per tile
turn), tells the board where the shaft really is. Moves are
**closed loop**: the board cuts power just before the target, early enough that the coast lands it
there. It learns how far the motor coasts, and it makes up to 3 small corrections if a move still
ends more than 0.5° off.

The encoder is optional. If calibration sees no encoder counts, the board falls back to the old
**timed** mode. That mode measures how long one tile turn takes, then keeps the motor on for
`angle / 360` of that time. It is only an estimate, because the motor coasts and the turn time
changes with load, supply voltage and temperature.

```
t74-motor/
  t74.py                  terminal tool: calibrate, move, home, goto, watch, interactive session
  t74_calibration.json    saved turn time, encoder counts per turn and speed (written by t74.py)
  firmware/
    main.cpp              the Uno sketch
    build_flash.sh        compile, and flash if you pass `upload`
```

## Wiring (one-off)

### Motor driver

| IBT-2 | Uno | |
|---|---|---|
| RPWM | D5 | forward (positive angles) |
| LPWM | D6 | reverse (negative angles) |
| R_EN | D7 | |
| L_EN | D8 | |
| VCC | 5V | IBT-2 logic |
| GND | GND | common ground, **required** |
| B+ / B- | motor supply | |
| M+ / M- | T74 motor | |

### Encoder: AMT10E2-V, 3 signal pins + power

| AMT10E2 connector | Uno | why this pin |
|---|---|---|
| **A** | **D2** | INT0: a hardware interrupt, so no edge is missed |
| **B** | **D3** | INT1: the Uno's only other hardware interrupt pin |
| **X** (index) | **D4** | pin-change interrupt. One pulse per encoder turn, used by `home` |
| **5V** | 5V | the encoder draws about 6 mA |
| **G** | GND | |

* **A and B must go to D2 and D3.** These are the only Uno pins with dedicated interrupts. The
  board counts every edge of both channels ("4x decoding"), so neither channel can be polled.
* **X is optional.** Without it everything works except `home` and `goto`. D4 is free and on the
  same port as A and B. On this motor-side mounting the index pulses about 20 times per tile turn,
  so `home` finds a repeatable *motor* position, not a unique tile angle (see below).
* **It doesn't matter which way round A and B go.** Calibration measures which way the count runs
  when the motor turns forward. If you swap A and B later, recalibrate (or the board stops with
  "counting the wrong way").
* The outputs are 5 V push-pull CMOS, so no resistors are needed. The sketch turns on the Uno's
  pull-ups anyway, so an unplugged encoder reads steady instead of floating.
* **Keep the encoder cable short and away from the motor wires.** The IBT-2 switches the motor
  current hard, and noise on A or B shows up as "missed edges" in `./t74.py watch`.

### Encoder resolution (DIP switch on the encoder)

**Set the switches to 500 PPR: `0 1 0 1` (1 = on).** The factory setting, 5120 PPR (all off), is
too fine for this mounting. The Uno can count roughly 50,000 edges per second. With the encoder on
the motor shaft, a tile turn of about 4 s means about 300 RPM at the encoder, which is about 100,000
edges per second at 5120 PPR. The Uno then misses edges and the count comes out wrong. One full
tile turn read about 415,871 counts at 5120 PPR, and that may already be an undercount.

At 500 PPR (2000 counts per encoder turn), one tile turn is about 40,600 counts (about 0.009° per
count), and the Uno sees about 10,000 edges per second. That leaves room for about 5× more speed.

`ENC_PPR` at the top of `main.cpp` must match the switches (it is set to 500). It is used for the
"encoder shaft" angle in `watch`, the turns-per-tile-turn figure printed at calibration, the homing guard,
and to detect that the index is on the motor side. Moves use the counts measured during
calibration.

If calibration prints `WARNING: ... missed encoder edges`, the encoder is still too fast. Lower the
PPR further (e.g. 250 PPR = `0 1 1 1`, then `ENC_PPR = 250`) or lower the speed.

### Other notes

* **Recommended: fit 10 kΩ pull-downs from R_EN and L_EN to GND.** Opening the serial port resets
  the Uno, and the pins float during the reset until `setup()` drives them LOW.
* If positive angles turn the wrong way, swap M+ and M-.
* The run speed is a PWM value from 1 to 255, set with `./t74.py speed <pwm>` (no reflash needed).
  With the encoder, changing the speed needs no recalibration. Without it, the turn time is cleared
  and you must recalibrate. Until you set one, the board uses `DEFAULT_PWM` at the top of `main.cpp`.

## Flash the firmware (one-off, or after editing main.cpp)

Several Unos are usually plugged into this PC, so find the right one first:

```bash
cd t74-motor
./t74.py list                                    # note the /dev/ttyACMx of the T74 board
cd firmware
./build_flash.sh                                 # build only, touches no hardware
PORT=/dev/ttyACMx ./build_flash.sh upload        # flash it (PORT is required)
cd ..                                            # back to t74-motor/ for the t74.py commands
```

From /ur-assembly/:

```bash
PORT=/dev/ttyACM3 ./t74-motor/firmware/build_flash.sh upload
```

If `t74.py list` shows more than one board, pass `--port /dev/ttyACMx` to every `t74.py`
command below. Close the Arduino IDE serial monitor first, because only one program can hold the
port.

## Test the encoder wiring (once, motor supply OFF)

```bash
./t74.py watch
```

Turn the shaft slowly by hand. You should see:

* the **count** change smoothly, in opposite directions for the two ways you turn it
* **missed edges** stay at 0. If it climbs, check the wiring and keep the cable away from the motor
  leads
* **index pulses** go up by one per full encoder shaft turn. If it stays at 0, check X on D4

Ctrl-C stops it.

## Every time you attach a tile

1. **Power off the motor supply.** Mount the tile and tighten it.
2. **Mark the tile.** Draw a clear line on its edge, and pick a fixed reference next to it, such
   as tape on the frame. Rotate the tile by hand until the mark lines up with the reference.
3. **Power on the motor supply.** Keep a hand near the keyboard: `s` + Enter or Ctrl-C stops it.
   If the motor only hums and doesn't turn, the speed is too low to get it started. Raise it, e.g.
   `./t74.py speed 80`, then try again.
4. **Calibrate over several turns:**
   ```bash
   ./t74.py calibrate --turns 5
   ```
   Press Enter to start. The motor turns forward slowly. Count the mark passing the reference, and
   press Enter again **the moment it comes back the 5th time**. The script divides by 5, so the
   error of your Enter press is 5× smaller than with one turn. The board saves the turn time and
   the encoder counts per tile turn, then measures how far the motor coasts. It also prints the
   encoder turns per tile turn, which is the gear ratio (expect about 20.3). The whole run must
   fit in 60 s. If it doesn't, use fewer turns or a higher speed.
5. **Optional, if you know the gear ratio exactly:** set the count directly to 4 × PPR × ratio.
   At 500 PPR that is 2000 × ratio, and keep the minus sign if calibration gave a negative number.
   ```bash
   ./t74.py counts 40500       # e.g. a 20.25:1 gearbox
   ```
6. **Check it:** `./t74.py move 90` four times should bring the mark back to the reference. Each
   move prints its own error, for example `error 0.07 deg`.
7. **Use it:**
   ```bash
   ./t74.py move 60      # +60 deg (forward), relative to where the tile is now
   ./t74.py move -30     # 30 deg back
   ```
   `move` takes 1 to 360 degrees, positive or negative.

The saved calibration and speed stay until you change them, so after a reboot or re-plug you can go
straight to step 7 **with the same tile**. With a new tile, start again at step 1.

**Opening the port resets the board, so it forgets the learned coast.** The first move of each
run may need one correction, which you will see as a "Correcting" line. For several moves in a row,
use the interactive session: it keeps one connection, so it keeps the learned coast.

**`home` and `goto` with the encoder on the motor side.** The index pulses once per *encoder*
turn, about 20 times per tile turn, so `home` stops at whichever of those comes first. That is a
repeatable motor position, not a unique tile angle, and the board says so after homing.
`./t74.py goto` therefore refuses, because each run would home to a different tile angle. Inside
one interactive session, `h` then `g 90` still works, with angles measured from where that session
homed. A unique tile zero needs a sensor on the tile side, such as a limit switch or a second index.

## Interactive session

```
$ ./t74.py
t74> c          start calibration (motor runs forward)
t74> m          mark is back at the reference: stop and save the turn time and counts
t74> k 40500    set encoder counts per tile turn
t74> h          home to the index pulse (0 deg)
t74> g 90       go to 90 deg from home
t74> 60         move +60 deg
t74> -30        move back 30 deg
t74> e          encoder count and angle
t74> p 80       set the speed to PWM 80
t74> s          STOP
t74> ?          board status
t74> help       list the commands
t74> q          quit (sends stop first)
```

## Why the script saves the calibration

Opening the serial port resets the Uno, and a reset erases everything in its memory. So `t74.py`
writes the speed, encoder counts per turn and turn time to `t74_calibration.json`. Each time it
connects, it sends them back to the board as `P<pwm>`, `K<counts>` and `T<ms>`.

## Serial protocol (115200 baud)

You can also type these straight into the Arduino IDE serial monitor. Any line ending works.

| send | effect |
|---|---|
| `C` | start calibration: motor runs forward |
| `M` | stop calibration, store the motor-on time and encoder counts for one turn, then measure the coast |
| `S` | stop now, motor disabled (works at any time) |
| `60`, `-30`, `15.5` | relative move, 1 to 360 degrees, negative = reverse |
| `H` | home: run forward to the index pulse, which becomes 0 deg |
| `G<deg>` | go to an absolute angle from home, e.g. `G90` (encoder only) |
| `E` | encoder count, encoder shaft angle, output shaft revolutions and angle, homed, index pulses, missed edges |
| `K<counts>` | load encoder counts per tile turn, signed, e.g. `K40500` |
| `T<ms>` | load a known turn time, e.g. `T4200` (500 to 60000 ms) |
| `P<pwm>` | set the run speed, e.g. `P80` (1 to 255). A different speed clears the turn time |
| `?` | print turn time, counts, speed and mode |

Safety limits in the firmware:

* A calibration that is still running after 60 s stops by itself and is discarded.
* No move or homing run can keep the motor on for more than 60 s.
* With the encoder, the motor stops with `ENCODER FAULT` in three cases: no counts for 1 s while
  the motor is on (broken wire or jammed motor), the count running the wrong way, or homing that
  turns more than 1.25 turns without seeing the index pulse.
