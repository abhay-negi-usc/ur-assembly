from glob import glob

from setuptools import find_packages, setup

package_name = 'ur_cable_pick_assemble_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhay',
    maintainer_email='negiabhay98@gmail.com',
    description=('Cable pick-and-ASSEMBLE for UR10e + 2F-85: picks the cable exactly as '
                 'ur_cable_pick_place_demo does, then assembles it (kinematic method first) with '
                 'a stand-off, compliance control, and a force-guarded chunked insertion.'),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cable_pick_assemble = ur_cable_pick_assemble_demo.cable_pick_assemble_node:main',
        ],
    },
)
