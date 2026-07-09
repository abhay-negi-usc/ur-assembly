from glob import glob

from setuptools import find_packages, setup

package_name = 'ur_assembly_demo'

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
    description='Fiducial-guided pick-and-assemble demo for UR10e + Robotiq 2F-85.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'assemble = ur_assembly_demo.assemble_node:main',
        ],
    },
)
