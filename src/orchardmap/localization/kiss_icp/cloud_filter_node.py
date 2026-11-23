#!/usr/bin/env python3
"""Drop ground (and near-field) points from /velodyne_points before KISS-ICP.

On flat ground every VLP-16 beam traces a ring at a fixed range around the sensor,
so the ground looks identical wherever the robot is. Inside an orchard row (little
along-row structure) point-to-point ICP gets pulled toward "no motion" by those
rings and under-estimates travel. Removing points below z_min (sensor frame) leaves
trunks/canopy, which do move with the robot. All fields (incl. per-point time) are kept.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


class CloudFilter(Node):
    def __init__(self):
        super().__init__('cloud_filter_node')
        self.declare_parameter('cloud_in', '/velodyne_points')
        self.declare_parameter('cloud_out', '/velodyne_points_filtered')
        self.declare_parameter('z_min', -0.45)    # m, sensor frame (sensor ~0.72 m above ground)
        self.declare_parameter('r_min', 0.0)      # m, horizontal near-field cut
        self.z_min = float(self.get_parameter('z_min').value)
        self.r_min = float(self.get_parameter('r_min').value)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(PointCloud2, self.get_parameter('cloud_out').value, qos)
        self.create_subscription(PointCloud2, self.get_parameter('cloud_in').value, self.cb, qos)

    def cb(self, msg):
        f = {x.name: x.offset for x in msg.fields}
        buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
        x, y, z = (buf[:, f[k]:f[k] + 4].copy().view(np.float32)[:, 0] for k in 'xyz')
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > self.z_min)
        if self.r_min > 0:
            keep &= (x * x + y * y) > self.r_min ** 2
        out = PointCloud2()
        out.header = msg.header
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.height = 1
        out.width = int(keep.sum())
        out.row_step = out.width * out.point_step
        out.is_dense = True
        out.data = buf[keep].tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(CloudFilter())


if __name__ == '__main__':
    main()
