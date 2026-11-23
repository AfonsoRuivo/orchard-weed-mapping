#!/usr/bin/env python3
"""Exit 0 if a ROS 2 node can be created (DDS participant comes up); used by the job queue."""
import rclpy

rclpy.init()
rclpy.create_node('ros_node_probe').destroy_node()
rclpy.shutdown()
print('ok')
