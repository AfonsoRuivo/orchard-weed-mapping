#!/usr/bin/env python3
"""Record any nav_msgs/Odometry topic to CSV: t,x,y,z,qx,qy,qz,qw (same format as
kissicp_odometry_recorder.py, so analysis/ate_vs_gps.py reads both).

    python3 odometry_recorder.py <out.csv> [topic]   (default topic /dlio/odom_node/odom)
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry


class Recorder(Node):
    def __init__(self, out_path, topic):
        super().__init__('odometry_recorder')
        self.f = open(out_path, 'w')
        self.f.write("t,x,y,z,qx,qy,qz,qw\n")
        self.n = 0
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(Odometry, topic, self.cb, qos)

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.f.write(f"{t:.9f},{p.x},{p.y},{p.z},{q.x},{q.y},{q.z},{q.w}\n")
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()


def main():
    rclpy.init()
    node = Recorder(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '/dlio/odom_node/odom')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.flush()
        node.f.close()


if __name__ == '__main__':
    main()
