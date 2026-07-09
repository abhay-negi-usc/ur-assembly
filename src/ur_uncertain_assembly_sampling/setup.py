from glob import glob

from setuptools import find_packages, setup

package_name = 'ur_uncertain_assembly_sampling'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml') + glob('config/*.csv')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abhay',
    maintainer_email='negiabhay98@gmail.com',
    description='Uncertain assembly sampling (perturbed chunked assembly + logging) for the UR10e.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uncertain_assembly_sampling ='
            ' ur_uncertain_assembly_sampling.uncertain_assembly_sampling_node:main',
        ],
    },
)
