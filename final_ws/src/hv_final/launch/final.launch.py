from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hv_final',
            executable='sign_vision',
            name='sign_vision_node',
            output='screen'
        ),
        Node(
            package='hv_final',
            executable='robot_control',
            name='robot_control_node',
            output='screen'
        )
    ])
