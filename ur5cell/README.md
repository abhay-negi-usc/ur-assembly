# ur5cell — the UR5 (CB3.1) cell

The UR5's apps, configs and tests, on top of the same `urlab` core the UR10e cell runs. Nothing
here is a copy of `urlab`: fixes to the core reach both cells.

| | |
|---|---|
| Arm | UR5 **CB3.1**, serial 2016353713 (built 2016-11-30), at **192.168.10.125** (PolyScope 3.15.8) |
| Connection | default mode — `ur_rtde` uploads its own script, like the UR10e. The External Control URCap (port 50002) must **not** be playing. `robot.use_ext_ur_cap: true` switches modes. |
| Gripper | Robotiq **2F-85** on a spacer, Modbus RTU from this PC over its USB-RS485 adapter (`gripper.type: robotiq_2f85`) |
| Camera | mounted on the spacer; driver not chosen (`camera.type: unset`) |
| F/T sensor | not mounted yet |

## What CB3 changes

`robot.series: cb3` (in `configs/ur5-robot.yaml`) switches `urlab/robot/arm.py`:

- **No wrist F/T sensor.** `getActualTCPForce()` is a joint-current *estimate*. Force guards are
  collision backstops only; compliance (admittance) waits for the external F/T sensor.
- **Tared in software.** `zero_ftsensor()` is e-Series-only, so `zero_ft()` remembers the current
  reading and `wrench()` subtracts it.
- **No force-mode gain scaling.** `force_mode()` skips it.
- **125 Hz RTDE** (`robot.rtde_frequency_hz`). Admittance tuning from the UR10e does not transfer:
  its servo loop actually runs ~2.65× faster than its configured rate.

## Configs

Every file is `ur5-` prefixed. A UR5 config names `robot_config: ur5-robot.yaml`, which carries the
machine **and** `frames_file: ur5-frames.yaml` — so it can never pick up the UR10e's hand-eye from
`configs/frames.yaml`. `tests/test_ur5_configs.py` checks that for every config here.

| file | what |
|---|---|
| `ur5-robot.yaml` | the machine: IP, series, connection mode, payload, bring-up speeds, frames file |
| `ur5-camera.yaml` | the camera driver (`camera.type`) — unset until the camera is chosen |
| `ur5-frames.yaml` | the tool stack as a chain (tool0 → spacer → camera / gripper → fingertip, grasp) and taught targets. `ur5-robot.yaml`'s `fingertip_grasp` / `grasp_tcp_offset` **name** these frames, so the numbers live here only |
| `ur5-monitor.yaml`, `ur5-cartesian.yaml`, `ur5-teleop.yaml` | bring-up configs for the generic `urlab` apps |

## Bring-up

```bash
python -m ur5cell.apps.monitor  [--wrench] [--csv f.csv]    # read-only
python -m ur5cell.apps.cartesian --dry-run
python -m ur5cell.apps.cartesian                             # slow, confirms each move
python -m ur5cell.apps.teleop                                # rigid jogging only
python -m pytest ur5cell/tests -q
```

`cartesian` jogs in `ur_base` (the pendant's Base axes) and `tool0` (its Tool axes), and teleop's
BASE mode in `ur_base` (`teleop.base_jog_frame`). urlab's `world` is ROS `base_link`, whose x/y
are the pendant's negated. Its R/P/Y moves turn about the
frame's x/y/z (UR's RX/RY/RZ), and each prompt says which axis.

**Run UR5 programs through `ur5cell.apps.*`, not `urlab.apps.*`.** Each wrapper sets
`$URLAB_CONFIG_DIR` to `ur5cell/configs` for that run and defaults `--config` to its `ur5-` config,
so it **cannot** load a UR10e config: `--config cartesian` (the UR10e's) is an error, not a run
against `robot.ip` .112. `python -m urlab.apps.cartesian` with no `--config` would load the UR10e's.
The same works by hand for anything else — `export URLAB_CONFIG_DIR=$PWD/ur5cell/configs` — but
unset it before UR10e work (`tests/` ignores it either way).

Before the first real move: PolyScope ≥ 3.3 (`ur_rtde`'s CB3 minimum), and a safety plane above the
bench on the pendant (Installation → Safety → Planes) — `urlab`'s ground-collision model is still
UR10e-only.

**Still to set as hardware arrives:** `reset.home_joints_deg` (teach it — until then do not run a
reset, which would fall back to a UR10e pose), `robot.payload` (spacer + gripper + camera,
unweighed), the spacer length and gripper clocking (`gripper` in `ur5-frames.yaml`, placeholders
at 0), the camera driver (`urlab/perception/camera.py` `CAMERAS`), the `ft_sensor` frame, and the
external F/T sensor path in `URArm.wrench()`.

## New task apps

Go in `ur5cell/apps/`. Start from `urlab/apps/behavior_demo.py`, call `ur5cell.use_ur5_configs()`
at the top of `main()`, and give `run_app` a `ur5-` config name — bare names then resolve in
`ur5cell/configs` only.
