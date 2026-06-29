from glob import glob

from setuptools import find_packages, setup

package_name = 'ur_admittance_demo'

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
    description='Admittance-control demo for a UR10e via the ros2_control admittance_controller.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'admittance_hold_demo = ur_admittance_demo.admittance_hold_demo:main',
        ],
    },
)
