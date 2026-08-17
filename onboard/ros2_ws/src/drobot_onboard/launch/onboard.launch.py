from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "manifest",
                default_value=EnvironmentVariable("DROBOT_MANIFEST", default_value=""),
                description="Absolute path to hardware/robot-runtime/four-leg.toml",
            ),
            DeclareLaunchArgument(
                "serial_port",
                default_value=EnvironmentVariable(
                    "DROBOT_SERIAL_PORT",
                    default_value="auto",
                ),
            ),
            DeclareLaunchArgument(
                "http_bind",
                default_value=EnvironmentVariable(
                    "DROBOT_HTTP_BIND",
                    default_value="0.0.0.0",
                ),
            ),
            DeclareLaunchArgument(
                "http_port",
                default_value=EnvironmentVariable(
                    "DROBOT_HTTP_PORT",
                    default_value="8080",
                ),
            ),
            DeclareLaunchArgument(
                "control_token",
                default_value=EnvironmentVariable(
                    "DROBOT_CONTROL_TOKEN",
                    default_value="",
                ),
            ),
            DeclareLaunchArgument("demo", default_value="false"),
            Node(
                package="drobot_onboard",
                executable="onboard_control",
                name="onboard_control",
                namespace="drobot",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "manifest_path": LaunchConfiguration("manifest"),
                        "serial_port": LaunchConfiguration("serial_port"),
                        "http_bind": LaunchConfiguration("http_bind"),
                        "http_port": ParameterValue(
                            LaunchConfiguration("http_port"),
                            value_type=int,
                        ),
                        "control_token": LaunchConfiguration("control_token"),
                        "demo": ParameterValue(
                            LaunchConfiguration("demo"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
