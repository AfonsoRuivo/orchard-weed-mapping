#!/usr/bin/env python3
"""Republish /velodyne_points with per-point times FAST-LIO can use for deskewing.

The velodyne driver stamps the header at the END of the sweep and stores per-point
`time` as a negative offset in seconds (about -0.1..0). FAST-LIO assumes the header
is the sweep START and offsets are >= 0, so it never deskewed these clouds.
This shifts header.stamp back to the earliest point and makes every offset >= 0
(still in seconds -> use timestamp_unit: 0 in the FAST-LIO config).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


class CloudTimeFix(Node):
    def __init__(self):
        super().__init__('cloud_time_fix_node')
        self.declare_parameter('in_topic', '/velodyne_points')
        self.declare_parameter('out_topic', '/velodyne_points_timefix')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(PointCloud2, self.get_parameter('out_topic').value, qos)
        self.sub = self.create_subscription(PointCloud2, self.get_parameter('in_topic').value, self.cb, qos)

    def cb(self, msg):
        f = {x.name: x.offset for x in msg.fields}
        if 'time' not in f:
            self.pub.publish(msg)
            return
        buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step).copy()
        o = f['time']
        t = buf[:, o:o + 4].copy().view(np.float32)[:, 0]
        x = buf[:, f['x']:f['x'] + 4].copy().view(np.float32)[:, 0]
        ok = np.isfinite(x) & np.isfinite(t)
        if not ok.any():
            return
        t0 = float(t[ok].min())
        t = np.where(ok, t - t0, 0.0).astype(np.float32)
        buf[:, o:o + 4] = t.view(np.uint8).reshape(-1, 4)
        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec + int(round(t0 * 1e9))
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)
        msg.data = buf.tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CloudTimeFix())


if __name__ == '__main__':
    main()
