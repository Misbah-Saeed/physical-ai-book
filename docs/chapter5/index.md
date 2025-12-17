---
title: "Gazebo Simulation"
sidebar_position: 5
---

## Overview

This chapter focuses on Gazebo, a powerful 3D robotics simulator widely used in Physical AI development. We will explore the differences between SDF and URDF, understand how Gazebo's physics engine works, learn about simulating various sensors, and practice spawning robots in a simulated environment.

## Learning Objectives

By the end of this chapter, you will be able to:
- Differentiate between URDF and SDF and understand their respective use cases.
- Comprehend the role of Gazebo's physics engine in robot simulation.
- Configure and simulate common sensors in Gazebo.
- Spawn robots into a Gazebo world from URDF/SDF models.
- Debug and interact with simulated robots.
- Apply your knowledge through practical exercises.

## SDF vs URDF

While both **URDF (Unified Robot Description Format)** and **SDF (Simulation Description Format)** are XML-based formats used to describe robots and environments, they serve different purposes within the ROS and Gazebo ecosystem.

| Feature             | URDF                                           | SDF                                                    |
| :------------------ | :--------------------------------------------- | :----------------------------------------------------- |
| **Primary Purpose** | Robot description (kinematics, dynamics, visuals) | Full simulation description (robots, environments, lights, sensors, physics, plugins) |
| **Scope**           | Single robot                                   | Multiple robots and the entire world                   |
| **Origin**          | ROS                                            | Gazebo                                                 |
| **Extensibility**   | Often extended with Xacro for modularity       | More comprehensive and designed for full simulation features |
| **Hierarchy**       | Tree structure                                 | Can represent acyclic graphs                           |
| **Sensors**         | Describes sensor mounts (physical presence)    | Describes sensor behavior, outputs, and simulation parameters |

**Key takeaway:** URDF describes *what a robot is*, while SDF describes *everything in a Gazebo simulation world*, including robots, static objects, lights, and sensor properties.

Gazebo can import URDF files, but it converts them internally to SDF to leverage the full range of simulation features. For complex Gazebo worlds, directly writing SDF is often preferred.

## Physics Engine

Gazebo integrates powerful physics engines (e.g., ODE, Bullet, DART, Simbody) to accurately simulate the interactions between rigid bodies, joints, and forces within the virtual environment. Key aspects include:

-   **Rigid Body Dynamics:** Calculates how objects move and rotate in response to forces and torques.
-   **Collision Detection:** Determines when objects in the simulation intersect, preventing them from passing through each other.
-   **Contact Resolution:** Applies forces to resolve collisions, simulating realistic bouncing or sliding.
-   **Joint Constraints:** Enforces the limits and types of motion defined by joints in the robot model.
-   **Gravity & Friction:** Environmental forces that affect all objects in the world.

Gazebo's physics engine settings (e.g., update rate, solver type, iterations) can be configured in the world's SDF file to balance simulation accuracy and computational performance.

## Sensors Simulation

Gazebo provides a robust framework for simulating a wide array of sensors, allowing developers to test perception algorithms without real hardware. This is achieved through **Gazebo plugins**, which are shared libraries loaded at runtime.

Commonly simulated sensors:

1.  **Cameras (RGB, Depth, RGB-D):**
    -   Plugins like `libgazebo_ros_camera.so` simulate realistic camera output, including noise, distortion, and publishing `sensor_msgs/Image` and `sensor_msgs/PointCloud2` messages to ROS 2 topics.

2.  **LiDAR / Laser Scanners:**
    -   Plugins such as `libgazebo_ros_laser.so` simulate 2D or 3D laser scans, accounting for range limits, angles, and noise. They publish `sensor_msgs/LaserScan` or `sensor_msgs/PointCloud2`.

3.  **IMU (Inertial Measurement Unit):**
    -   `libgazebo_ros_imu_sensor.so` provides realistic acceleration, angular velocity, and orientation data, publishing `sensor_msgs/Imu` messages.

4.  **Contact Sensors:**
    -   `libgazebo_ros_bumper.so` detects collisions between specified links and publishes `std_msgs/Bool` or custom contact messages.

**Example Sensor Plugin (SDF snippet for a camera):**

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box>0.01 0.01 0.01</box>
    </geometry>
  </visual>
  <inertial>
    <mass>0.1</mass>
    <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
  </inertial>
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>30.0</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>true</ros>
      <cameraName>main_camera</cameraName>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
      <point_cloud_cutoff>0.5</point_cloud_cutoff>
      <point_cloud_cutoff_max>5.0</point_cloud_cutoff_max>
      <point_cloud_cutoff_min>0.0</point_cloud_cutoff_min>
      <ros_topic>image_raw</ros_topic>
      <ros_depth_topic>depth/image_raw</ros_depth_topic>
      <ros_point_cloud_topic>depth/points</ros_point_cloud_topic>
    </plugin>
  </sensor>
</link>
```

## Spawn Robot Examples

Spawning a robot into a Gazebo world typically involves launching Gazebo with a world file and then using a ROS 2 launch file or command-line tools to load and place the robot model.

### 1. Launching an Empty Gazebo World

First, start Gazebo:

`gazebo --verbose empty.world`
or, more commonly with ROS 2:
`ros2 launch gazebo_ros gazebo.launch.py`

### 2. Spawning a URDF Robot with `spawn_entity.py`

Assuming you have a `my_robot.urdf` file:

```bash
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file install/my_robot_description/share/my_robot_description/urdf/my_robot.urdf -x 0 -y 0 -z 1.0
```

-   `-entity`: Name of the spawned model in Gazebo.
-   `-file`: Absolute path to your URDF/SDF file.
-   `-x`, `-y`, `-z`: Initial position of the robot.

### 3. Using a ROS 2 Launch File for Spawning

This is the most common and robust method. You would create a launch file that:
1.  Launches Gazebo.
2.  Loads the robot description (URDF/Xacro) to the ROS parameter server.
3.  Launches `robot_state_publisher` to publish joint states as TF transformations.
4.  Launches `spawn_entity.py` (or a similar Gazebo ROS plugin) to insert the model into the simulation.

Example `my_robot_spawn.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Get path to your robot description package
    robot_description_package = FindPackageShare('my_robot_description')
    robot_description_path = PathJoinSubstitution(
        [robot_description_package, 'urdf', 'my_robot.urdf.xacro'] # Use xacro if you have one
    )

    # Convert xacro to URDF if needed, or just load URDF
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ', robot_description_path
    ])

    # Publish the robot description to the ROS parameter server
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}]
    )

    # Spawn the robot into Gazebo
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # Launch Gazebo itself
    gazebo_launch_path = PathJoinSubstitution([
        FindPackageShare('gazebo_ros'), 'launch', 'gazebo.launch.py'
    ])
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path),
        launch_arguments={'world': 'empty.world'}.items()
    )

    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher_node,
        spawn_entity_node,
    ])
```

## Exercises & Quizzes

1.  **Question:** Explain a scenario where using SDF directly to describe an entire Gazebo world would be more advantageous than trying to model everything using multiple URDFs and separate static models.
2.  **Code Task:** Create a simple Gazebo world SDF file (`my_simple_world.sdf`) that includes a flat ground plane, a light source, and a static box obstacle. Launch Gazebo with this world.
3.  **Simulation Task:** Take the URDF for the two-link arm you created in Chapter 4, and create a ROS 2 launch file that:
    -   Launches an empty Gazebo world.
    -   Loads your URDF into the `robot_description` parameter.
    -   Spawns your two-link arm robot into the Gazebo world.
    -   Launches `joint_state_publisher_gui` to control the arm's joints manually.
4.  **Question:** Describe how Gazebo simulates sensor data for an RGB-D camera. What kind of ROS 2 messages would you expect such a simulated sensor to publish?
5.  **Multiple Choice:** Which of the following is primarily responsible for calculating how objects interact physically in a Gazebo simulation?
    -   A) URDF Parser
    -   B) ROS 2 Topic Communication
    -   C) Physics Engine
    -   D) Visualization Tools
