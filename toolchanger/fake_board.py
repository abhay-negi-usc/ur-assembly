#!/usr/bin/env python3
"""A fake toolchanger board on a pty, so toolchanger.py can be tested with no hardware.

    ./fake_board.py          run the driver against the simulation and assert the protocol

Opening the real port pulls DTR, resets the Arduino and swings the servo, so this exists to
exercise the driver without moving anything. It speaks exactly the lines main.cpp prints.
"""

import os
import pty
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolchanger import ToolChanger  # noqa: E402

THRESH = 100          # must match main.cpp
RAW_TOOL = 800        # what the sensor reads with a tool in front of it
RAW_EMPTY = 20        # ... and with nothing there


def board(fd, tool_present):
    """Mimic main.cpp: one ASCII byte in, one line out."""
    status = 0
    motor = False

    def raw():
        return RAW_TOOL if tool_present[0] else RAW_EMPTY

    def confirm():
        #  the board confirms only when the sensor agrees with the commanded status
        if tool_present[0] == (status > 0):
            os.write(fd, f"{status} confirmed!\r\n".encode())
        else:
            os.write(fd, f"emergency stop raw={raw()} thresh={THRESH}\r\n".encode())

    while True:
        try:
            c = os.read(fd, 1)
        except OSError:
            return
        if not c:
            return
        ch = c.decode('ascii', errors='replace')
        if ch == 's':
            confirm()
        elif ch == 'r':
            tool = 'yes' if tool_present[0] else 'no'
            os.write(fd, f"raw {raw()} thresh {THRESH} tool {tool} status {status}\r\n".encode())
        elif ch == 'm':
            motor = not motor
            os.write(fd, b"Motor On\r\n" if motor else b"Motor Off\r\n")
        elif ch.isdigit():
            new = int(ch)
            if new != status:
                os.write(fd, f"changed status from {status} to {new}\r\n".encode())
                status = new
            confirm()
        #  anything else (line endings, noise) is ignored, as on the board


def main():
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
        #  releasing while the sensor still sees the tool must trip the emergency stop
        assert tc.release() is False

        tool_present[0] = False  # the tool is physically gone
        assert tc.release() is True
        assert tc.probe() == RAW_EMPTY
        #  holding with nothing in the gripper must trip it the other way
        assert tc.hold() is False
    finally:
        tc.close()

    print('\nALL DRIVER ASSERTIONS PASSED')


if __name__ == '__main__':
    main()
