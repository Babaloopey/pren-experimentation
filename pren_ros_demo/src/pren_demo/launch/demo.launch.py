"""Full PREN dog digital twin: sim + EKF + fake perception + mission + RViz.

    ros2 launch pren_demo demo.launch.py [rviz:=false] [n_bones:=2] [click_adds:=true] [seed:=42] [voice_miss_rate:=0.2]

keyboard_teleop needs a real terminal -> start it separately (run/keyboard.sh).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('pren_demo')
    arg = LaunchConfiguration

    def node(exe, **params):
        return Node(package='pren_demo', executable=exe, name=exe, output='screen',
                    parameters=[params] if params else [])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('n_bones', default_value='1'),
        DeclareLaunchArgument('click_adds', default_value='false'),
        DeclareLaunchArgument('seed', default_value='-1'),
        DeclareLaunchArgument('voice_miss_rate', default_value='0.0'),

        # --- simulation only (replaced by hardware drivers on the robot) ---
        node('loopback_sim'),
        node('bone_detector',
             n_bones=ParameterValue(arg('n_bones'), value_type=int),
             click_adds=ParameterValue(arg('click_adds'), value_type=bool),
             seed=ParameterValue(arg('seed'), value_type=int)),
        node('world_viz'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_truth_tf',
             arguments=['--x', '0.2', '--z', '0.25',
                        '--frame-id', 'base_truth', '--child-frame-id', 'camera_truth']),
        node('command_recognizer',
             voice_miss_rate=ParameterValue(arg('voice_miss_rate'), value_type=float)),

        # --- identical on the robot ---
        Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
             output='screen', parameters=[os.path.join(share, 'config', 'ekf.yaml')]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_tf',
             arguments=['--x', '0.2', '--z', '0.25',
                        '--frame-id', 'base_link', '--child-frame-id', 'camera_link']),
        node('field'),
        node('dog_brain'),
        node('search_node'),
        node('cmd_mux'),

        Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
             arguments=['-d', os.path.join(share, 'config', 'demo.rviz')],
             condition=IfCondition(arg('rviz'))),
    ])
