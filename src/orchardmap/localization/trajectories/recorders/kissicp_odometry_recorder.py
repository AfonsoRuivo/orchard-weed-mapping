#!/usr/bin/env python3
"""Record /kiss/odometry to CSV: t,x,y,z,qx,qy,qz,qw (matches imu_lidar_rotation.py's --odom format)."""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry


class Recorder(Node):
    def __init__(self, out_path):
        super().__init__('kissicp_odometry_recorder')
        self.f = open(out_path, 'w')
        self.f.write("t,x,y,z,qx,qy,qz,qw\n")
        self.n = 0
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(Odometry, '/kiss/odometry', self.cb, qos)

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.f.write(f"{t:.9f},{p.x},{p.y},{p.z},{q.x},{q.y},{q.z},{q.w}\n")
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()
            self.get_logger().info(f"recorded {self.n} odometry msgs")


def main():
    out_path = sys.argv[1]
    rclpy.init()
    node = Recorder(out_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.flush()
        node.f.close()
        node.get_logger().info(f"final count: {node.n}")


if __name__ == '__main__':
    main()
