# ur_vision_demo

A computer-vision demo for **ROS2 Jazzy**: connects to one or more **Intel RealSense** cameras
and streams **ArUco fiducial marker poses with respect to each camera**.

Per camera it publishes:
- **tf2 transforms** `<camera>_color_optical_frame → <camera>_marker_<id>`
- a **`geometry_msgs/PoseArray`** on `/<camera>/aruco_poses`
- an annotated **debug image** on `/<camera>/debug_image`

Multiple cameras run side-by-side (one RealSense node + one detector per camera). The marker
**dictionary** and **size** are set in [`config/cameras.yaml`](config/cameras.yaml).

## How it works

Each detector subscribes to its camera's `color/image_raw` + `color/camera_info`, runs
`cv2.aruco` detection, and estimates each marker's pose with
`cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)` from the marker's known physical size and the
camera intrinsics. Pose is in the **color optical frame** (Z forward out of the lens). The node
handles both the old and new OpenCV ArUco APIs.

## Prerequisites

```bash
# RealSense ROS2 wrapper + cv_bridge + tf helpers
sudo apt install ros-jazzy-realsense2-camera ros-jazzy-cv-bridge ros-jazzy-tf-transformations
```
`cv2.aruco` is needed. Ubuntu's `python3-opencv` normally includes it; if `import cv2.aruco`
fails, add `pip install opencv-contrib-python`.

> Docker: the cameras are USB devices, so the container needs USB access — start it with
> `--device-cgroup-rule='c 81:* rmw' -v /dev:/dev` (or `--privileged`), like the gripper's
> serial device. Verify inside the container with `rs-enumerate-devices -s`.

### Find your camera serials and set the config
```bash
rs-enumerate-devices -s        # lists each device's serial number
```
Edit [`config/cameras.yaml`](config/cameras.yaml): one entry per camera with its `serial_no`
(**required when using more than one camera**), and set the ArUco `dictionary` /
`marker_size_m` (the printed black square's side length, in meters — measure it).

### Print markers
Generate markers from the **same dictionary** you configured, e.g. with OpenCV:
```python
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
cv2.imwrite('marker_0.png', cv2.aruco.generateImageMarker(d, 0, 600))
```
Print, measure the actual square size, and put that in `marker_size_m`.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_vision_demo --symlink-install
source install/setup.bash

ros2 launch ur_vision_demo aruco_demo.launch.py
```

Watch the poses:
```bash
# per-camera pose array
ros2 topic echo /camera1/aruco_poses

# tf: distance/orientation of a marker w.r.t. the camera
ros2 run tf2_ros tf2_echo camera1_color_optical_frame camera1_marker_0

# annotated image
ros2 run rqt_image_view rqt_image_view /camera1/debug_image
```
In RViz, set Fixed Frame to a camera optical frame and add **TF** + a **PoseArray** display on
`/camera1/aruco_poses`.

> Topic check: this assumes RealSense publishes at `/<camera>/camera/color/image_raw` (the
> modern `camera_namespace`/`camera_name` layout). If `ros2 topic list | grep image_raw` shows
> a different path, set `image_topic`/`camera_info_topic` on the detector to match.

## Running modes

- **Detectors only** (cameras already running elsewhere):
  `ros2 launch ur_vision_demo aruco_demo.launch.py launch_cameras:=false`
- **Custom config path:**
  `ros2 launch ur_vision_demo aruco_demo.launch.py config_file:=/path/to/cameras.yaml`
- **One detector by hand** against any image topic:
  ```bash
  ros2 run ur_vision_demo aruco_pose_node --ros-args \
    -p image_topic:=/camera1/camera/color/image_raw \
    -p camera_info_topic:=/camera1/camera/color/camera_info \
    -p aruco_dictionary:=DICT_5X5_100 -p marker_size_m:=0.04 \
    -p marker_frame_prefix:=cam1_marker_
  ```

## Node parameters (`aruco_pose_node`)

| Parameter | Default | Description |
|---|---|---|
| `image_topic` | `color/image_raw` | Color image topic (relative to the node namespace). |
| `camera_info_topic` | `color/camera_info` | Camera intrinsics topic. |
| `aruco_dictionary` | `DICT_4X4_50` | Predefined cv2.aruco dictionary name. |
| `marker_size_m` | `0.05` | Printed marker square side length (m). |
| `marker_frame_prefix` | `marker_` | tf child-frame prefix; set per camera to keep frames unique. |
| `camera_frame` | `""` | Override the parent frame; empty = use `camera_info` frame_id. |
| `publish_debug_image` | `true` | Publish the annotated `debug_image`. |

## Notes

- **Marker size and dictionary must match the printed markers** or poses will be wrong /
  undetected. Wrong size → correct orientation but scaled (wrong) distance.
- Pose accuracy depends on the camera intrinsics from `camera_info` (RealSense factory
  calibration is usually fine) and marker print quality/flatness.
- For multiple cameras, set every `serial_no` — without it, two nodes can fight over one
  device.
- This streams marker pose **in the camera frame**. To express markers in a robot/world frame,
  publish a static transform from the robot base to each `*_color_optical_frame` (hand-eye
  calibration) and let tf compose them.
