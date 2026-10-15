# ur5cell — the UR5 (CB3.1) cell

The UR5's apps, configs and tests, on top of the same `urlab` core the UR10e cell runs. Nothing
here is a copy of `urlab`: fixes to the core reach both cells.

| | |
|---|---|
| Arm | UR5 **CB3.1**, serial 2016353713 (built 2016-11-30), at **192.168.10.127** |
| Connection | default mode — `ur_rtde` uploads its own script, like the UR10e. The External Control URCap (port 50002) must **not** be playing. `robot.use_ext_ur_cap: true` switches modes. |
| Gripper / F/T sensor / camera | not mounted yet — `gripper.type: none`, `camera.type: unset` |

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
| `ur5-frames.yaml` | tool0-attached frames and taught targets — hand-eye is an identity **placeholder** |
| `ur5-monitor.yaml`, `ur5-cartesian.yaml`, `ur5-teleop.yaml` | bring-up configs for the generic `urlab` apps |

## Bring-up

```bash
python -m urlab.apps.monitor   --config ur5cell/configs/ur5-monitor.yaml            # read-only
python -m urlab.apps.cartesian --config ur5cell/configs/ur5-cartesian.yaml --dry-run
python -m urlab.apps.cartesian --config ur5cell/configs/ur5-cartesian.yaml          # slow, confirms each move
python -m urlab.apps.teleop    --config ur5cell/configs/ur5-teleop.yaml             # rigid jogging only
python -m pytest ur5cell/tests -q
```

Before the first real move: PolyScope ≥ 3.3 (`ur_rtde`'s CB3 minimum), and a safety plane above the
bench on the pendant (Installation → Safety → Planes) — `urlab`'s ground-collision model is still
UR10e-only.

**Still to set as hardware arrives:** `reset.home_joints_deg` (teach it — until then do not run a
reset, which would fall back to a UR10e pose), `robot.payload`, the gripper and camera drivers
(`urlab/robot/gripper.py` `GRIPPERS`, `urlab/perception/camera.py` `CAMERAS`), the `ft_sensor` and
`fingertip` frames, and the external F/T sensor path in `URArm.wrench()`.

## New task apps

Go in `ur5cell/apps/`. Start from `urlab/apps/behavior_demo.py`, and give `run_app` an absolute
default config — `run_app('...', ur5cell.config_path('ur5-<app>'), build_and_run)` — since a bare
config name is looked up in the UR10e's `configs/`.
