from glob import glob

from setuptools import setup

package_name = 'pren_demo'

NODES = [
    'loopback_sim',
    'field',
    'bone_detector',
    'command_recognizer',
    'dog_brain',
    'search_node',
    'cmd_mux',
    'keyboard_teleop',
    'world_viz',
]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenz',
    maintainer_email='abo@gb-netz.de',
    description='PREN1 dog robot digital twin',
    license='MIT',
    entry_points={
        'console_scripts': [f'{n} = {package_name}.{n}:main' for n in NODES],
    },
)
