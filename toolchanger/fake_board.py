#!/usr/bin/env python3
"""A fake toolchanger board on a pty, so toolchanger.py can be tested with no hardware.

    ./fake_board.py          run the driver against the simulation and assert the protocol

Opening the real port pulls DTR, resets the Arduino and swings the servo, so this exists to
exercise the driver without moving anything. It speaks exactly the lines firmware/ prints.
"""

import os
import pathlib
import pty
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolchanger import ToolChanger, ToolChangerError  # noqa: E402

THRESH = 100          # must match coupler.cpp
RAW_TOOL = 800        # what the sensor reads with a tool in front of it
RAW_EMPTY = 20        # ... and with nothing there


def board(fd, tool_present):
    """Mimic firmware/: one ASCII byte in (plus a number after 'd'), one line out."""
    #  setup() ends with `status = checkTool() ? 1 : 0; changeServo(status > 0);` -- a board that
    #  boots with anything in front of the probe comes up believing it is ALREADY LOCKED. That
    #  matters because changeStatus() only swings the servo when the status CHANGES, so a hold()
    #  in that state confirms without moving. Starting at 0 regardless made this simulation
    #  unable to reproduce the one failure mode that has actually bitten: a pick that reported a
    #  grip it never made.
    status = 1 if tool_present[0] else 0
    held = status > 0     # toolHeld: locked onto a tool the sensor actually saw
    motor = False
    bypassed = False
    dc = 0                # DC motor percent

    #  setup() announces itself once, which is what the driver syncs on. Delayed a beat
    #  because the driver opens the pty after this thread starts, and a banner written
    #  before it is listening is a banner it never sees.
    time.sleep(0.2)
    os.write(fd, b"toolchanger ready\r\n")

    def raw():
        return RAW_TOOL if tool_present[0] else RAW_EMPTY

    def emergency():
        os.write(fd, f"emergency stop raw={raw()} thresh={THRESH}\r\n".encode())

    def confirm():
        os.write(fd, f"{status} confirmed!\r\n".encode())

    while True:
        try:
            c = os.read(fd, 1)
        except OSError:
            return
        if not c:
            return
        ch = c.decode('ascii', errors='replace')

        if ch == 's':
            os.write(fd, b"tool present\r\n" if tool_present[0] else b"tool absent\r\n")
            #  only a tool that WAS gripped and has gone is an emergency
            confirm() if (not held or tool_present[0]) else emergency()
        elif ch == 'r':
            tool = 'yes' if tool_present[0] else 'no'
            byp = 'on' if bypassed else 'off'
            os.write(fd, f"raw {raw()} thresh {THRESH} tool {tool} status {status} "
                         f"bypass {byp}\r\n".encode())
        elif ch == 'b':
            bypassed = not bypassed
            os.write(fd, b"sensor bypass ON -- grip is NOT verified\r\n" if bypassed
                         else b"sensor bypass off\r\n")
        elif ch == 'm':
            motor = not motor
            os.write(fd, b"Motor On\r\n" if motor else b"Motor Off\r\n")
        elif ch == 'd':
            #  Serial.parseInt(): an optional '-', digits, anything else ends it; nothing
            #  parses as 0, i.e. stop. Constrained to -100..100 as in dc_motor.cpp.
            text = b''
            while True:
                b = os.read(fd, 1)
                if not b or not (b.isdigit() or (b == b'-' and not text)):
                    break
                text += b
            try:
                dc = int(text)
            except ValueError:
                dc = 0
            dc = max(-100, min(100, dc))
            os.write(fd, f"dc motor {dc}%\r\n".encode())
        elif ch.isdigit():
            new = int(ch)
            if new != status:
                os.write(fd, f"changed status from {status} to {new}\r\n".encode())
                status = new
            if bypassed:
                #  no sensor opinion is sought when bypassed
                held = False
                os.write(fd, b"(sensor bypassed)\r\n")
                confirm()
            elif status > 0:
                #  locking onto nothing is a NOTICE, not an emergency -- the servo still moved
                held = tool_present[0]
                if held:
                    confirm()
                else:
                    os.write(fd, f"locked, no tool detected raw={raw()} "
                                 f"thresh={THRESH}\r\n".encode())
            else:
                held = False
                #  releasing always succeeds; the tool sitting there afterwards is normal
                here = 'still in the changer' if tool_present[0] else 'gone'
                os.write(fd, f"released, tool {here}\r\n".encode())
                confirm()
        #  anything else (line endings, noise) is ignored, as on the board


def check_docs_match_firmware():
    """Fail if the servo angles quoted in the driver or the README have drifted from coupler.cpp.

    coupler.cpp is the only place the angles actually take effect; everywhere else is prose that
    silently goes stale when they are retuned. They were swapped once with the docs left
    behind, which is exactly the sort of thing nobody notices until the tool drops."""
    here = pathlib.Path(__file__).parent
    firmware = (here / 'firmware' / 'coupler.cpp').read_text()

    lock = int(re.search(r'const int lockAngle\s*=\s*(\d+)', firmware).group(1))
    nolock = int(re.search(r'const int noLockAngle\s*=\s*(\d+)', firmware).group(1))

    driver = (here / 'toolchanger.py').read_text()
    readme = (here / 'README.md').read_text()

    quoted = [
        # (what it should be, pattern, where)
        (nolock, r'0 = unlocked \(servo (\d+) deg\)', 'toolchanger.py'),
        (lock, r'>0 = locked \(servo (\d+) deg\)', 'toolchanger.py'),
        (lock, r'lockAngle, (\d+) deg', 'toolchanger.py'),
        (nolock, r'noLockAngle, (\d+) deg', 'toolchanger.py'),
        (lock, r'servo to \*\*(\d+) deg\*\* \(locked\)', 'README.md'),
        (nolock, r'0 unlocks \(servo (\d+) deg\)', 'README.md'),
        (lock, r'>0 locks \((\d+) deg\)', 'README.md'),
    ]

    for expected, pattern, where in quoted:
        text = driver if where == 'toolchanger.py' else readme
        found = re.search(pattern, text)
        assert found, f"{where}: nothing matched {pattern!r} -- the wording moved, fix this check"
        actual = int(found.group(1))
        assert actual == expected, (
            f"{where} says {actual} deg where coupler.cpp says {expected} "
            f"(lockAngle={lock}, noLockAngle={nolock})")

    print(f"angles agree everywhere: locked {lock} deg, unlocked {nolock} deg")


def main():
    check_docs_match_firmware()

    master, slave = pty.openpty()
    tool_present = [True]  # what the proximity sensor "sees"; flip it mid-test
    threading.Thread(target=board, args=(master, tool_present), daemon=True).start()

    tc = ToolChanger(os.ttyname(slave), settle=0.1, timeout=3, verbose=True)
    try:
        assert tc.hold() is True, 'hold with a tool present should confirm'
        assert tc.status() is True
        assert tc.probe() == RAW_TOOL, 'probe should report the mounted reading'
        assert tc.motor() is True, 'first toggle turns the motor on'
        assert tc.motor() is False

        #  the DC motor: signed speed, and what comes back is what the BOARD says it is doing
        assert tc.drive(60) == 60
        assert tc.drive('-40') == -40, 'a reversal, from a string as the CLI passes it'
        assert tc.stop() == 0
        for bad in (150, -101, '6.5', 'fast'):
            try:
                tc.drive(bad)
            except ToolChangerError:
                pass
            else:
                raise AssertionError(f'drive({bad!r}) should be refused before anything is sent')

        #  THE regression this guards: the tool is still sitting in the changer after a
        #  release, which is normal and must NOT come back as an emergency stop
        assert tc.release() is True, 'release with the tool still present must confirm'
        assert tc.status() is True, 'released + tool present is a normal state'

        tool_present[0] = False  # the tool is physically pulled away
        assert tc.release() is True
        assert tc.probe() == RAW_EMPTY
        #  locking onto nothing still moves the servo and is only a notice -- but hold() is
        #  False, because a pick that missed must not lift
        assert tc.hold() is False, 'hold with nothing to grip must not read as a grip'
        assert tc.status() is True, 'a coupler locked empty on purpose is not an emergency'

        #  the real emergency: a tool that WAS gripped falls out
        tc.release()
        tool_present[0] = True
        assert tc.hold() is True
        tool_present[0] = False
        assert tc.status() is False, 'a gripped tool that has gone must alarm'
        tc.release()

        #  ... unless the sensor is bypassed, which is the point of the bypass: work the
        #  servo while the probe is untrustworthy, without the board pretending it checked
        assert tc.bypass() is True, 'first toggle turns the bypass on'
        assert tc.hold() is True, 'bypassed, hold confirms without consulting the sensor'
        assert tc.release() is True
        assert tc.bypass() is False, 'second toggle turns it back off'
        assert tc.hold() is False, 'un-bypassed, an empty lock is reported again'
    finally:
        tc.close()

    print('\nALL DRIVER ASSERTIONS PASSED')


if __name__ == '__main__':
    main()
