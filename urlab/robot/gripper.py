"""Robotiq 2F-85 over Modbus RTU -- replaces the ros2_robotiq_gripper driver + its controllers.

COUNTS ARE NATIVE HERE. The ROS stack exposed the gripper as a joint in RADIANS, so the grasp
check had to convert rad -> counts through `full_close_rad` (a value the config still marks
"TODO: verify me", because it was a fudge factor nobody could measure directly). The gripper's
actual register interface is 0-255 counts in both directions. The conversion, and the calibration
constant it depended on, are simply gone -- `closed_counts: 228` is now compared against a number
the hardware reports.

The status register also carries gOBJ, the object-detection bits, which the ROS action never
surfaced. That is what makes EMPTY-pickup detection possible for the first time (see
`grasp_result`) -- previously only the cable-outside-the-groove failure was detectable.

Register map (Robotiq 2F, Modbus RTU, slave 9):
    write 0x03E8: [action | 0 | 0 | position | speed | force]   -- 3 uint16 words
    read  0x07D0: [status | 0 | fault | pos_req | pos | current]
"""

import time

import numpy as np

from .. import log as urlog

log = urlog.get('gripper')

_WRITE_ADDR = 0x03E8
_READ_ADDR = 0x07D0
_SLAVE = 9

# gOBJ, bits 6-7 of the status byte.
OBJ_MOVING = 0
OBJ_STOPPED_OPENING = 1     # something blocked the fingers while opening
OBJ_STOPPED_CLOSING = 2     # something blocked the fingers while closing -- an object is held
OBJ_AT_TARGET = 3           # reached the requested position, nothing in the way


class GripperError(RuntimeError):
    pass


class Robotiq2F85:
    """Blocking gripper control in counts (0 = fully open, 255 = fully closed)."""

    def __init__(self, cfg):
        g = cfg.section('gripper')
        self.port = g.get('port', '/dev/ttyUSB0')
        self.baud = int(g.get('baud', 115200))
        self.open_counts = int(g.get('open_counts', 0))
        self.closed_counts = int(g.get('closed_counts', 255))
        self.speed = int(g.get('speed_counts', 255))
        self.force = int(g.get('force_counts', 150))
        self.timeout_s = float(g.get('timeout_s', 5.0))
        self.dry_run = bool(cfg.get_path('robot.dry_run', False))

        if self.dry_run:
            log.warning('DRY RUN: gripper commands are logged, not sent.')
            self.client = None
            self._sim_pos = self.open_counts
            return

        try:
            from pymodbus.client import ModbusSerialClient
        except ImportError as exc:
            raise GripperError('pymodbus is not installed. `pip install pymodbus`') from exc

        # pymodbus renamed the slave-id keyword across 3.x (unit -> slave -> device_id), so we
        # discover the one THIS version accepts on first use rather than hardcoding it. See _call.
        self._unit_kw = None
        self.client = ModbusSerialClient(
            port=self.port, baudrate=self.baud, bytesize=8, parity='N', stopbits=1, timeout=0.2)
        if not self.client.connect():
            raise GripperError(
                f'Cannot open {self.port}. Check the device exists (ls /dev/ttyUSB*), that you '
                f'are in the dialout group, and that no other process holds it. Prefer a stable '
                f'/dev/serial/by-id/... path -- ttyUSB numbering changes on replug.')
        self.activate()

    # ------------------------------------------------------------------ raw io
    def _call(self, method_name, *args, **kwargs):
        """Call a pymodbus client method, passing the slave id under whatever keyword this version
        wants. pymodbus renamed it across 3.x (unit -> slave -> device_id) AND removed the old
        names, so a hardcoded keyword raises `unexpected keyword argument`. We try the known names
        in order, cache the one that works, and reuse it thereafter."""
        fn = getattr(self.client, method_name)
        candidates = [self._unit_kw] if self._unit_kw else ['slave', 'device_id', 'unit']
        last = None
        for kw in candidates:
            try:
                result = fn(*args, **kwargs, **{kw: _SLAVE})
            except TypeError as exc:
                if 'unexpected keyword' not in str(exc):
                    raise                        # a real signature error, not the id-keyword rename
                last = exc
                continue
            self._unit_kw = kw                   # cache the working keyword
            return result
        raise GripperError(
            'Could not find the slave-id keyword for this pymodbus version (tried slave, '
            f'device_id, unit). Installed pymodbus may be too new/old. Last error: {last}')

    def _write(self, action, position, speed, force):
        words = [(action << 8) | 0x00,
                 (0x00 << 8) | int(np.clip(position, 0, 255)),
                 (int(np.clip(speed, 0, 255)) << 8) | int(np.clip(force, 0, 255))]
        result = self._call('write_registers', _WRITE_ADDR, words)
        if result.isError():
            raise GripperError(f'Modbus write failed: {result}')

    def _read(self):
        result = self._call('read_holding_registers', _READ_ADDR, count=3)
        if result.isError():
            raise GripperError(f'Modbus read failed: {result}')
        # Robotiq input registers (read from 0x07D0): each 16-bit register packs two status bytes,
        # MSB first. regs[0]: [gripper status | reserved]; regs[1]: [FAULT (gFLT) | position-request
        # echo (gPR)]; regs[2]: [POSITION (gPO) | current (gCU)]. The fault and the actual position
        # are the HIGH bytes of regs[1]/regs[2] -- reading the low bytes instead picks up the gPR
        # echo (which equals the last commanded position, e.g. 0xFF after a full close) and the
        # motor current, which is why a close-to-255 looked like "fault 0xFF".
        regs = result.registers
        status = regs[0] >> 8
        return {
            'activated': bool(status & 0x01),               # gACT
            'status': (status >> 4) & 0x03,                 # gSTA: 3 = activation complete
            'obj': (status >> 6) & 0x03,                    # gOBJ
            'fault': regs[1] >> 8,                          # gFLT (Byte 2); 0 = no fault
            'pos_req': regs[1] & 0xFF,                       # gPR -- echo of the commanded position
            'pos': regs[2] >> 8,                            # gPO -- ACTUAL POSITION IN COUNTS
            'current': regs[2] & 0xFF,                       # gCU -- motor current
        }

    # ------------------------------------------------------------------ lifecycle
    def activate(self):
        """Activate the gripper (required once after power-up; the fingers do a full stroke)."""
        if self.dry_run:
            return True
        state = self._read()
        if state['activated'] and state['status'] == 3:
            log.info('Gripper already activated.')
            return True

        log.info('Activating the gripper (the fingers will make a full stroke)...')
        self._write(0x00, 0, 0, 0)                          # clear rACT
        time.sleep(0.1)
        self._write(0x01, 0, self.speed, self.force)        # set rACT
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = self._read()
            if state['status'] == 3:
                log.info('Gripper activated.')
                return True
            time.sleep(0.2)
        raise GripperError('Gripper did not finish activating within 10 s.')

    def clear_fault(self):
        """Clear a gripper FAULT by RE-ACTIVATING it (clear rACT, then set rACT) -- the only way the
        Robotiq clears a latched fault. This RE-HOMES the fingers (a full open/close stroke), so any
        held object is RELEASED. Returns True once re-activated with no fault, False on timeout."""
        if self.dry_run:
            self._sim_pos = 0
            return True
        log.warning('Clearing gripper fault via re-activation -- the fingers will re-home (any held '
                    'object is released).')
        self._write(0x00, 0, 0, 0)                          # clear rACT
        time.sleep(0.5)
        self._write(0x01, 0, self.speed, self.force)        # set rACT -> re-activate
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = self._read()
            if state['status'] == 3 and not state['fault']:
                log.info('Gripper fault cleared; re-activated.')
                return True
            time.sleep(0.2)
        log.error('Gripper did not re-activate / clear the fault within 10 s (fault 0x%02X).',
                  self._read()['fault'])
        return False

    def disconnect(self):
        if self.client is not None:
            self.client.close()

    # ------------------------------------------------------------------ state
    def position(self):
        """Current finger position in COUNTS (0 open .. 255 closed)."""
        if self.dry_run:
            return self._sim_pos
        return self._read()['pos']

    def fault(self):
        """Current fault code (gFLT); 0 = no fault. A non-zero fault latches until clear_fault()."""
        if self.dry_run:
            return 0
        return self._read()['fault']

    def object_detected(self):
        """True if the fingers stalled on something rather than reaching the commanded position."""
        if self.dry_run:
            return False
        return self._read()['obj'] in (OBJ_STOPPED_OPENING, OBJ_STOPPED_CLOSING)

    # ------------------------------------------------------------------ motion
    def go_to(self, counts, label='', wait=True):
        """Command a position in counts and (by default) block until the fingers settle.

        A STALL IS SUCCESS. The fingers stopping early because they met an object is the normal,
        desired outcome of a grasp -- it is not an error, and this returns True for it. Only a
        comms failure or a timeout is a failure."""
        counts = int(np.clip(counts, 0, 255))
        log.info('--> gripper %s (%d counts)', label or 'move', counts)
        if self.dry_run:
            self._sim_pos = counts
            return True

        self._write(0x09, counts, self.speed, self.force)   # rACT | rGTO
        if not wait:
            return True

        deadline = time.monotonic() + self.timeout_s
        time.sleep(0.05)                                    # let gOBJ leave "moving"
        while time.monotonic() < deadline:
            state = self._read()
            if state['fault']:
                log.error('Gripper fault 0x%02X.', state['fault'])
                return False
            if state['obj'] != OBJ_MOVING:
                log.info('    settled at %d counts (obj=%d).', state['pos'], state['obj'])
                return True
            time.sleep(0.02)
        log.error('Gripper did not settle within %.1f s.', self.timeout_s)
        return False

    def open(self, label='open'):
        return self.go_to(self.open_counts, label)

    def close(self, label='close'):
        return self.go_to(self.closed_counts, label)

    def go_to_fraction(self, frac, label=''):
        """0.0 = open, 1.0 = closed, interpolating between the configured endpoints."""
        frac = float(np.clip(frac, 0.0, 1.0))
        counts = self.open_counts + frac * (self.closed_counts - self.open_counts)
        return self.go_to(round(counts), label or f'{frac * 100:.0f}% closed')

    # ------------------------------------------------------------------ grasp check
    def grasp_result(self, groove_counts, empty_counts, faces_max_counts, tolerance=1,
                     detect_empty=True):
        """Classify a completed close as 'ok' | 'missed' | 'empty' from the finger POSITION.

        More obstruction = LESS closed, so the three states order by position:

          * pos <= faces_max_counts (~223)  -- the cable is caught on the flat FACES, not in the
                                               groove: it props the fingers open. FAILED ('missed').
          * pos >= empty_counts - tolerance (~228) -- the fingers closed FULLY: nothing is in the
                                               groove. EMPTY (only when detect_empty).
          * otherwise (~groove_counts, 225) -- the cable is seated in the groove. SUCCESS ('ok').

        Position-based (not gOBJ): with these fingertips a seated cable (225) and an empty close
        (228) differ by position, so empty is reliably separable -- unlike the previous fingertips
        where both reached full closure and only the UNVERIFIED gOBJ bit could tell them apart."""
        if self.dry_run:
            return 'ok'
        state = self._read()
        pos, obj = state['pos'], state['obj']

        if pos <= faces_max_counts:
            log.warning('Grasp MISSED: %d <= %d counts -- the cable is on the fingertip FACES, not '
                        'in the groove.', pos, faces_max_counts)
            return 'missed'
        if detect_empty and pos >= empty_counts - tolerance:
            log.warning('Grasp EMPTY: %d counts -- the fingers closed fully, no cable in the '
                        'groove.', pos)
            return 'empty'
        log.info('Grasp OK: %d counts (obj=%d) -- cable seated in the groove (~%d).',
                 pos, obj, groove_counts)
        return 'ok'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False
