# monitor — live end-effector + frame readout

A diagnostic (no ROS equivalent). Prints the joints plus **tool0, fingertip, camera, and grasp**
frames — all in `base_link`, `xyz` (mm) + `rpy` (deg, extrinsic XYZ) — updating in place, while you
**drive the arm by hand**.

```
joints (deg): [ +90.00, -135.00, -135.00,  +0.00, +90.00,  +0.00]
  base_link <- tool0       : xyz=[ ... ] mm  rpy=[ ... ] deg
  base_link <- fingertip   : xyz=[ ... ] mm  rpy=[ ... ] deg
  base_link <- camera1...  : xyz=[ ... ] mm  rpy=[ ... ] deg
  base_link <- grasp       : xyz=[ ... ] mm  rpy=[ ... ] deg
```

**Read-only, so pendant freedrive just works.** It connects with `RTDEReceiveInterface` **only** — it
uploads no control script and needs no Remote Control, unlike the demos (which *drive* the arm). So:

1. Put the robot in **Freedrive on the pendant** (LOCAL control).
2. Run the monitor.
3. Push the arm around — the poses update live.

The frames are built from the **same config sections the demos use** (`hand_eye` → camera,
`fingertip_grasp` → fingertip, `grasp_tcp_offset` → grasp), so what you read here is exactly what a
demo would command. That makes this the tool to **measure the config values the hardware checklist
leaves open** (`assembly.target`, a grasp pose, `touch.contact_z_offset_m`): freedrive to the spot,
read the frame off here, paste it into the config.

```bash
python -m urlab.apps.monitor
python -m urlab.apps.monitor --config cable_pick_place --rate 20
python -m urlab.apps.monitor --wrench --gripper     # + TCP wrench (N, Nm) and gripper counts
python -m urlab.apps.monitor --csv poses.csv        # append every sample to a CSV
python -m urlab.apps.monitor --freedrive            # SOFTWARE freedrive via teachMode
```

Config: any demo config (for `robot.ip` + the frame offsets); defaults to `cable_pick_assemble`.

| flag | meaning |
|---|---|
| `--config <name>` | which config supplies `robot.ip` and the tool0→frame offsets |
| `--robot-ip <ip>` | override `robot.ip` |
| `--rate <hz>` | refresh rate (default 10) |
| `--freedrive` | enable **software** freedrive (`teachMode`) — needs **Remote Control**; drive by hand without the pendant button |
| `--wrench` | also show the TCP wrench (**uncompensated** — reads the tool weight, since read-only can't set the payload) |
| `--gripper` | also show gripper position in counts (opens the Modbus port; activates the gripper only if not already active) |
| `--csv <path>` | append `t`, joints, every frame, and (if enabled) wrench/gripper to a CSV |
| `--once` | print one sample and exit |

Caveats:

- **Assumes the pendant TCP is the all-zeros tool0** (the urlab convention). If a non-zero TCP is set
  on the pendant, `getActualTCPPose` reports *that*, and every frame here is offset by it.
- `--wrench` is uncompensated; for true external force use a demo that sets `robot.payload`.
- `--freedrive` needs Remote Control (it uploads a control script); the default read-only mode does
  not — which is why the default coexists with pendant freedrive.
