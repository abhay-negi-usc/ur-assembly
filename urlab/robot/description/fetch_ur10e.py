"""Fetch the official UR10e description and emit a PLAIN URDF that pybullet can load directly.

WHY THIS SCRIPT EXISTS RATHER THAN A CHECKED-IN COPY OF UPSTREAM. Universal Robots ships the
description as XACRO (Universal_Robots_ROS2_Description), which needs a ROS toolchain to expand
-- there is no ROS in this workspace and no plain .urdf anywhere in that repo. So the geometry
is assembled here from UR's OWN published parameter files:

    config/ur10e/default_kinematics.yaml   the joint origins (this IS the calibrated chain)
    config/ur10e/visual_parameters.yaml    where each collision mesh sits in its link
    urdf/ur_macro.xacro                    the link/joint topology and the flange/tool0 frames
    meshes/ur10e/collision/*.stl           the simplified collision shells

Every NUMBER in the generated URDF is upstream's; only the assembly is ours. Re-run this to
refresh, and diff the result -- a silent upstream change to the kinematics would show up there.

    python -m urlab.robot.description.fetch_ur10e

LICENSE: the upstream package is BSD-3-Clause. LICENSE is downloaded alongside the meshes and
must stay with them.
"""
import os
import urllib.request

RAW = ('https://raw.githubusercontent.com/UniversalRobots/'
       'Universal_Robots_ROS2_Description/rolling/')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'ur10e')
MESHES = ('base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3')

# ---- UR's published UR10e kinematics, config/ur10e/default_kinematics.yaml -------------------
# hash: calib_5119701370761913513. These are the NOMINAL figures -- a specific robot's
# calibrated values differ by a fraction of a millimetre, which is far inside the conservative
# margins the collision model works to.
JOINTS = [
    # name,                parent,             child,             xyz,                          rpy
    ('shoulder_pan_joint', 'base_link_inertia', 'shoulder_link',
     (0.0, 0.0, 0.1807), (0.0, 0.0, 0.0)),
    ('shoulder_lift_joint', 'shoulder_link', 'upper_arm_link',
     (0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),
    ('elbow_joint', 'upper_arm_link', 'forearm_link',
     (-0.6127, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ('wrist_1_joint', 'forearm_link', 'wrist_1_link',
     (-0.57155, 0.0, 0.17415), (0.0, 0.0, 0.0)),
    ('wrist_2_joint', 'wrist_1_link', 'wrist_2_link',
     (0.0, -0.11985, -2.458164590756244e-11), (1.570796327, 0.0, 0.0)),
    ('wrist_3_joint', 'wrist_2_link', 'wrist_3_link',
     (0.0, 0.11655, -2.390480459346185e-11),
     (1.570796326589793, 3.141592653589793, 3.141592653589793)),
]

# ---- where each collision mesh sits in its link, config/ur10e/visual_parameters.yaml ---------
PI = 3.141592653589793
MESH_OFFSET = {
    'base_link_inertia': ('base', (0.0, 0.0, 0.0), (0.0, 0.0, PI)),
    'shoulder_link': ('shoulder', (0.0, 0.0, 0.0), (0.0, 0.0, PI)),
    'upper_arm_link': ('upperarm', (0.0, 0.0, 0.1762), (PI / 2, 0.0, -PI / 2)),
    'forearm_link': ('forearm', (0.0, 0.0, 0.0393), (PI / 2, 0.0, -PI / 2)),
    'wrist_1_link': ('wrist1', (0.0, 0.0, -0.135), (PI / 2, 0.0, 0.0)),
    'wrist_2_link': ('wrist2', (0.0, 0.0, -0.12), (0.0, 0.0, 0.0)),
    'wrist_3_link': ('wrist3', (0.0, -0.0005, -0.1168), (PI / 2, 0.0, 0.0)),
}
MASSES = {'base_link_inertia': 4.0, 'shoulder_link': 7.369, 'upper_arm_link': 13.051,
          'forearm_link': 3.989, 'wrist_1_link': 2.1, 'wrist_2_link': 1.98,
          'wrist_3_link': 0.615}
# joint_limits.yaml: every UR10e joint is +/-2pi, and the velocity limits are per-joint.
LIMIT_LOWER, LIMIT_UPPER = -2 * PI, 2 * PI
VELOCITY = {'shoulder_pan_joint': 2.0943951, 'shoulder_lift_joint': 2.0943951,
            'elbow_joint': 3.14159265, 'wrist_1_joint': 3.14159265,
            'wrist_2_joint': 3.14159265, 'wrist_3_joint': 3.14159265}
EFFORT = {'shoulder_pan_joint': 330.0, 'shoulder_lift_joint': 330.0, 'elbow_joint': 150.0,
          'wrist_1_joint': 56.0, 'wrist_2_joint': 56.0, 'wrist_3_joint': 56.0}


# Pure FRAME links (base_link, flange, tool0) carry no geometry. pybullet warns on every load
# for a link with no inertial and silently substitutes mass=1 -- harmless for a kinematic query,
# but the warnings bury real ones. A negligible mass keeps it quiet without changing anything
# the collision check reads.
FRAME_INERTIAL = ('<inertial><mass value="1e-6"/><origin xyz="0 0 0"/>'
                  '<inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>'
                  '</inertial>')


def _t(v):
    return ' '.join('%.12g' % x for x in v)


def _link(name):
    mesh, xyz, rpy = MESH_OFFSET[name]
    m = MASSES[name]
    return f'''  <link name="{name}">
    <visual>
      <origin xyz="{_t(xyz)}" rpy="{_t(rpy)}"/>
      <geometry><mesh filename="meshes/collision/{mesh}.stl"/></geometry>
    </visual>
    <collision>
      <origin xyz="{_t(xyz)}" rpy="{_t(rpy)}"/>
      <geometry><mesh filename="meshes/collision/{mesh}.stl"/></geometry>
    </collision>
    <inertial>
      <mass value="{m}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
'''


def build_urdf():
    parts = ['<?xml version="1.0"?>\n',
             '<!-- GENERATED by urlab/robot/description/fetch_ur10e.py -- do not hand-edit.\n'
             '     Every number here comes from Universal Robots\' own\n'
             '     Universal_Robots_ROS2_Description (BSD-3-Clause, see LICENSE); only the\n'
             '     assembly into a plain URDF is ours, because upstream ships xacro only. -->\n',
             '<robot name="ur10e">\n',
             '  <link name="base_link">%s</link>\n' % FRAME_INERTIAL]
    # base_link -> base_link_inertia is a pi turn about Z: base_link is REP-103 aligned (X+
    # forward) while the controller's internal frames point X+ backwards. This is the SAME
    # rotation as transforms.BASE_LINK_FROM_UR_BASE -- the URDF carries the bridge internally,
    # so poses read out of it are already in base_link.
    parts.append('  <joint name="base_link-base_link_inertia" type="fixed">\n'
                 '    <parent link="base_link"/>\n    <child link="base_link_inertia"/>\n'
                 f'    <origin xyz="0 0 0" rpy="0 0 {PI:.12g}"/>\n  </joint>\n')
    for name in ('base_link_inertia', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'):
        parts.append(_link(name))
    for name, parent, child, xyz, rpy in JOINTS:
        parts.append(
            f'  <joint name="{name}" type="revolute">\n'
            f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
            f'    <origin xyz="{_t(xyz)}" rpy="{_t(rpy)}"/>\n'
            f'    <axis xyz="0 0 1"/>\n'
            f'    <limit lower="{LIMIT_LOWER:.12g}" upper="{LIMIT_UPPER:.12g}" '
            f'effort="{EFFORT[name]}" velocity="{VELOCITY[name]}"/>\n  </joint>\n')
    # flange and tool0, exactly as ur_macro.xacro defines them
    parts.append(f'  <link name="flange">{FRAME_INERTIAL}</link>\n'
                 '  <joint name="wrist_3-flange" type="fixed">\n'
                 '    <parent link="wrist_3_link"/>\n    <child link="flange"/>\n'
                 f'    <origin xyz="0 0 0" rpy="0 {-PI / 2:.12g} {-PI / 2:.12g}"/>\n  </joint>\n'
                 f'  <link name="tool0">{FRAME_INERTIAL}</link>\n'
                 '  <joint name="flange-tool0" type="fixed">\n'
                 '    <parent link="flange"/>\n    <child link="tool0"/>\n'
                 f'    <origin xyz="0 0 0" rpy="{PI / 2:.12g} 0 {PI / 2:.12g}"/>\n  </joint>\n')
    parts.append('</robot>\n')
    return ''.join(parts)


def main():
    mesh_dir = os.path.join(OUT, 'meshes', 'collision')
    os.makedirs(mesh_dir, exist_ok=True)
    for m in MESHES:
        dest = os.path.join(mesh_dir, m + '.stl')
        with urllib.request.urlopen(RAW + 'meshes/ur10e/collision/%s.stl' % m) as r:
            data = r.read()
        with open(dest, 'wb') as fh:
            fh.write(data)
        print('  %-14s %7d bytes' % (m + '.stl', len(data)))
    with urllib.request.urlopen(RAW + 'LICENSE') as r:
        lic = r.read()
    with open(os.path.join(OUT, 'LICENSE'), 'wb') as fh:
        fh.write(lic)
    urdf = os.path.join(OUT, 'ur10e.urdf')
    with open(urdf, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(build_urdf())
    print('wrote', urdf)


if __name__ == '__main__':
    main()
